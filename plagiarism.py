from transformers import AutoTokenizer, BloomForCausalLM
import torch, json
import tqdm as tqdm
import chromadb
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Arguments used in user interface for plagiarims checker')

parser.add_argument('model', type=str, help='Path to model (Model D2)')
parser.add_argument('tokenizer', type=str, help='Path to tokenizer (tokenizer_sme)')
parser.add_argument('database', type=str, help='Path to database (chroma)')
parser.add_argument('--collection', type=str, default='test', help='Choose collection in database (test)')

from text_extractors import Extractor

class PlagiarismChecker():
    def __init__(self, model_path, tokenizer_path, database_path, tresh=0.05):
        """
        Initilize Plagiarism checker

        Arguments:
        model_path: path to sami model to produce word embeddings
        tokenizer_path: path to sami tokenizer
        database_path: path to chromaDB
        """

        self.model = BloomForCausalLM.from_pretrained(
            model_path,
            output_hidden_states=True,
            cache_dir="/raid/rpa020/"
        ).to('cuda')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.client = chromadb.PersistentClient(database_path)
        self.tresh = tresh

    def _set_collection(self, collection_choice):
        self.collection = self.client.get_or_create_collection(collection_choice, metadata={"hnsw:space": "cosine"})

    def _normalize(self, vector):
        """ Normalizes a vector to unit length using L2 norm. """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _s2v(self, sentence):
        """
        Sentence to vector via word embeddings produced by sami model

        Arguments:
        sentence: sentence to feed model

        Returns: Word embeddings dim(1, 2064)
        """

        input_ids = self.tokenizer(sentence, return_tensors='pt').input_ids.to('cuda')

        with torch.no_grad():
            outputs = self.model(input_ids)
            hidden_states = outputs.hidden_states

        last_hidden_state = hidden_states[-1]

        attention_mask = self.tokenizer(sentence, return_tensors='pt').attention_mask.to('cuda')
        masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
        sentence_embedding = torch.mean(masked_hidden_state, dim=1)

        return sentence_embedding.cpu().numpy()


    
    def _cosine_search(self, sentence, n_results):
        """
        Vector search with cosine as similarity function

        Arguments:
        sentence: sentence to convert to vector to search for
        n_results: retreieve the n closest vectors

        Returns: query results
        """

        vec = self._s2v(sentence)
        vec = self._normalize(vec)

        res = self.collection.query(query_embeddings=vec[0].tolist(), n_results=n_results)
        return res

    def extract_data(self, document_path, file_type):
        """
        Deserialization.

        Arguments:
        document_path -> data/a_scraped_raw_data/sme/file_name
        file_type -> txt/docx/xml/html/pdf (pdf might contain alot of garbage)

        Returns: JSON object from document
        """

        if not isinstance(document_path, str):
            raise Exception("document_path must be a string")

        e = Extractor()

        if file_type == 'txt':
            return e.extract_from_txt([document_path], 'json')
        elif file_type == 'docx':
            return e.extract_from_docx([document_path], 'json')
        elif file_type == 'xml':
            return e.extract_from_xml([document_path], 'json')
        elif file_type == 'pdf':
            return e.extract_from_pdf([document_path], 'json')
        elif file_type == 'html':
            return e.extract_from_html([document_path], 'json')
        else:
            raise Exception("File type not supported for plagiarism check")

    
    def plagiarism_checker(self, data, n_results=10):
        """
        Checks for plagiarism in the data library

        Arguments:
        data -> JSON object returned from PlagiarismChecker.extract_data
        n_results -> give top 10 similarity results (as plagiarism in one sentence might occur from several sources)

        Returns: Plagiarism score
        """

        d = {}
        d_plag = {}

        for instance in data[0][0]['content']:
            res = self._cosine_search(instance, n_results=n_results)

            for i in range(0, n_results):
                cosine = res['distances'][0][i]
                similarity = 1 - cosine
                
                if similarity > 0.95:
                    source = res['metadatas'][0][i]['Source']

                    try:
                        d[source].append(similarity)
                    except KeyError:
                        d[source] = d.get(source, [similarity])
        for key in d:
            score = sum(d[key])/len(data[0][0]['content'])
            
            if score > self.tresh:
                d_plag[key] = f"{(score*100):.2f}%"

        print(d_plag)
        return d_plag
        

    def add_entry(self, URL, sentence):
        """ 
        Add entry to the plagiarism library 
        
        Arguments: 
        URL -> data path to document
        sentence -> Sentence to store

        Return: None
        """
        vec = self._s2v(sentence)
        vec = self._normalize(vec)
        
        metadata = {"Source": URL, "Sentence": sentence}
        
        self.collection.add(
            embeddings=vec[0].tolist(),
            metadatas=metadata,
            ids=self.count()
        )

    def add_data(self, data):
        """ 
        Add document to the plagirasim checker. 
        
        Arguments:
        data -> JSON object retrieved from PlagiarismChecker.extract_data
        
        Return: None
        """
        
        vectors = []
        metas = []
        idx = []

        i = self.count()
        for instance in data[0]:
            for content in instance['content']:
                vec = self._s2v(content)
                vec = self._normalize(vec)

                metadata = {"Source": instance["url"], "Sentence": content}
                
                vectors.append(vec[0].tolist())
                metas.append(metadata)
                idx.append(f"{i}")

                i += 1

                if (i % 10000) == 0:
                    self.collection.add(
                        embeddings=vectors,
                        metadatas=metas,
                        ids=idx
                    )
                    vectors = []
                    metas = []
                    idx = []

        self.collection.add(
            embeddings=vectors,
            metadatas=metas,
            ids=idx
        )
        vectors = []
        metas = []
        idx = []

    def delete_entry(self, ids):
        """ Delete an entry """
        self.collection.delete(ids=ids)

    def delete_all(self, collection_name):
        """ Delete all entries in a collection """
        self.client.delete_collection(collection_name)
        
    def count(self):
        """ Returns vector count """
        return self.collection.count()
    
    def list_collections(self):
        """ List all collections in the client """
        return self.client.list_collections()

def main():

    args = parser.parse_args()

    model_path = args.model
    tokenizer_path = args.tokenizer
    database_path = args.database
    collection_name = args.collection

    p = PlagiarismChecker(model_path, tokenizer_path, database_path)
    set_collection = 0

    for collection in p.list_collections():
        if collection_name == collection.name:
            p._set_collection(collection_name)
            set_collection = 1
        
    if set_collection == 1:
        jdata = p.extract_data("testdata/100_sample.txt", 'txt')
        p.plagiarism_checker(jdata)

        jdata = p.extract_data("testdata/50_sample.txt", 'txt')
        p.plagiarism_checker(jdata)

        jdata = p.extract_data("testdata/10_sample.txt", 'txt')
        p.plagiarism_checker(jdata)

        jdata = p.extract_data("testdata/one_line_sample.txt", 'txt')
        p.plagiarism_checker(jdata)

        jdata = p.extract_data("testdata/0_sample.txt", 'txt')
        p.plagiarism_checker(jdata)
        
    else:
        raise Exception(f"Collection does not exist. Only these collections exists: {p.list_collections()}")


if __name__ == '__main__':
    main()

