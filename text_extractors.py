import os
import requests
import numpy as np
from bs4 import BeautifulSoup
import pickle, json
from tqdm import tqdm
from os.path import join as pjoin
import re
import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
import docx # python-docx



# Utils
from termcolor import cprint
import logging
import datetime

cp = lambda msg,clr='green': cprint(msg, clr)

# Set up logging
def get_logger(log_file_name):
    """
    Example:
    logger = get_logger("Log file name")        # Log file name is the log file's prefix
    logger.info("info message")
    logger.warning("a warning")
    logger.error("error message")
    """
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%I-%M-%S %p")
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(f'{__name__}_{log_file_name}')
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(f'logs/{log_file_name}_{formatted_time}.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    return logger


class Extractor():
    """
    Extract text from url (local or web)
    File types: html, pdf, txt, docx, xml
    Automatic encoding detection, extracting text in Unicode
    """
    def __init__(self, logger=None):
        self.logger = logger
        self.north_sami_char_all = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Á', 'á', 'Č', 'č', 'Đ', 'đ', 'Ŋ', 'ŋ', 'Š', 'š', 'Ŧ', 'ŧ', 'Ž', 'ž']
        self.north_sami_char_additional = ['Á', 'á', 'Č', 'č', 'Đ', 'đ', 'Ŋ', 'ŋ', 'Š', 'š', 'Ŧ', 'ŧ', 'Ž', 'ž']
    def _delete_lines_not_containing_chars(self, lines):
        """ Delete lines not containing any North Sami characters
            # lines: list of strings """
        char_list = self.north_sami_char_all
        return [line for line in lines if any(char in line for char in char_list)]
    def _normalize_spaces(self, string_list):
        """ Replace one or more whitespace characters (\s+) with a single space """
        return [re.sub(r'\s+', ' ', string) for string in string_list]
    def save_to_binary_file(self, data, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    def load_from_binary_file(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    def save_to_txt_file(self, text, file_path):
        with open(file_path, 'w') as file:
            file.write(text)
    def save_to_json_file(self, text, file_path):
        with open(file_path, 'w') as file:
            json.dump(text, file, ensure_ascii=False)
    def extract_text(self, filetype, urls, texttype='json'):
        if filetype == ".html":
            return self.extract_from_html(urls, texttype)
        elif filetype == ".txt":
            return self.extract_from_txt(urls, texttype)
        elif filetype == ".docx":
            return self.extract_from_docx(urls, texttype)
        elif filetype == ".xml":
            return self.extract_from_xml(urls, texttype)
        elif filetype == ".pdf":
            return self.extract_from_pdf(urls, texttype)
    def _read_file(self, file_path, encoding='utf-8'):
        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            html_content = file.read()
        return html_content
    def _extract_encoding(self, html_content, file_path):
        # To find the charset in a meta tag
        pattern = r'charset=["\']?([\w-]+)["\']?'
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            self.logger.error(f"Could not find encoding in {file_path}")
            return 'utf-8'
    def _read_file_with_detected_encoding(self, file_path):
        """
        Detect file encoding by reading the charset in meta tag. If not found, use default utf-8 with errors ignored.
        """
        html_content = self._read_file(file_path)
        detected_encoding = self._extract_encoding(html_content, file_path)
        html_content = self._read_file(file_path, encoding=detected_encoding)
        unicode_content = html_content.encode(detected_encoding).decode(detected_encoding)      # converting to byte-iso-8859-1 then to unicode
        return unicode_content, detected_encoding
    def extract_from_html(self, urls, texttype):
        all_url_json = []
        all_text = ""
        encoding_set = set()
        for url in tqdm(urls):
            try:
                if os.path.isfile(url):     # local content
                    content, detected_encoding = self._read_file_with_detected_encoding(url)
                    encoding_set.add(detected_encoding)
                else:       # web content
                    import requests
                    response = requests.get(url)
                    content = response.content
                soup = BeautifulSoup(content, 'html.parser')
                # Remove all script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                visible_text = soup.get_text()
                paragraphs = visible_text.split('\n')
                # Remove empty paragraphs and strip leading/trailing spaces
                paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
                # Accumulate the paragraphs
                paragraphs = self._delete_lines_not_containing_chars(paragraphs)
                paragraphs = self._normalize_spaces(paragraphs)
                all_url_json.append({"url": url, "content": paragraphs})
                all_text += "\n".join(paragraphs) + "\n\n"
            except Exception as e:
                self.logger.error(f"Error in {url}: {e}")
                continue
        if texttype == 'txt': return all_text, encoding_set
        elif texttype == 'json': return all_url_json, encoding_set
    def _join_broken_words(self, lines):
        """
        Join words broken by hyphenation at end of line
        """
        lines = [line for line in lines if line.strip()]
        i = 0
        while i < len(lines)-1:
            if lines[i].endswith('-'):
                split_words = lines[i+1].split()
                lines[i] = lines[i][:-1] + split_words[0]
                lines[i+1] = ' '.join(split_words[1:])
            i += 1
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        return lines
    
    def _carbage_collection(self, lines):
        lines = [line for line in lines if line.strip()]

        i = 0
        while i < len(lines)-1:
            if ('' in lines[i] or '�' in lines[i]):
                lines.remove(lines[i])
            else:
                space = 0
                for char in lines[i]:
                    if char.isspace():
                        space += 1

                single_char = len(lines[i]) - space
                
                if space in range(single_char - 1, single_char + 1):
                    lines.remove(lines[i])
        return lines

    def extract_from_pdf(self, urls, texttype):
        all_url_json = []
        all_text = ""
        for url in tqdm(urls):
            try:
                if os.path.isfile(url):
                    doc = fitz.open(url)
                    print(url)
                else:
                    raise FileNotFoundError(f"No file found at {url}")
                url_full_text = []
                for page in doc:
                    blocks = page.get_text("blocks")
                    # https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_text
                    # https://pymupdf.readthedocs.io/en/latest/recipes-text.html
                    for block in blocks:
                        if not "<image:" in block[4]:   # ignore images
                            text = block[4]             # 4th element contains text
                            lines = text.split('\n')
                            lines = self._join_broken_words(lines)
                            # Join all lines in a block
                            block_text = ' '.join(line.strip() for line in lines)
                            url_full_text.append(block_text)

                url_full_text = self._delete_lines_not_containing_chars(url_full_text)
                url_full_text = self._normalize_spaces(url_full_text)
                self._carbage_collection(url_full_text)
                all_text += "\n".join(url_full_text) + "\n\n"
                all_url_json.append({"url": url, "content": url_full_text})
            except Exception as e:
                self.logger.error(f"Error in {url}: {e}")
                continue
        if texttype == 'txt': return all_text, None
        elif texttype == 'json': return all_url_json, None
        
    def extract_from_xml(self, urls, texttype):
        def _recursive_extract(element):
            #recursively extract text from each element
            text = []
            if element.text and element.text.strip():
                text.append(element.text.strip())
            for subelement in element:
                text.extend(_recursive_extract(subelement))
                if subelement.tail and subelement.tail.strip():
                    text.append(subelement.tail.strip())
            return text
        all_url_json = []
        all_text = ""
        for url in tqdm(urls):
            try:
                if os.path.isfile(url):
                    tree = ET.parse(url)
                    root = tree.getroot()
                else:
                    raise FileNotFoundError(f"No file found at {url}")
                url_full_text = _recursive_extract(root)
                url_full_text = self._delete_lines_not_containing_chars(url_full_text)
                url_full_text = self._normalize_spaces(url_full_text)
                all_text += "\n".join(url_full_text) + "\n\n"
                all_url_json.append({"url": url, "content": url_full_text})
            except Exception as e:
                self.logger.error(f"Error in {url}: {e}")
                continue
        if texttype == 'txt': return all_text, None
        elif texttype == 'json': return all_url_json, None
    def extract_from_docx(self, urls, texttype):
        all_url_json = []
        all_text = ""
        for url in tqdm(urls):
            try:
                if os.path.isfile(url):
                    doc = docx.Document(url)
                else:
                    raise FileNotFoundError(f"No file found at {url}")
                url_full_text = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
                url_full_text = self._delete_lines_not_containing_chars(url_full_text)
                url_full_text = self._normalize_spaces(url_full_text)
                all_text += "\n".join(url_full_text) + "\n\n"
                all_url_json.append({"url": url, "content": url_full_text})
            except Exception as e:
                self.logger.error(f"Error in {url}: {e}")
                continue
        if texttype == 'txt': return all_text, None
        elif texttype == 'json': return all_url_json, None
    def extract_from_txt(self, urls, texttype):
        all_url_json = []
        all_text = ""
        for url in tqdm(urls):
            try:
                if os.path.isfile(url):
                    with open(url, 'r') as infile:
                        url_full_text = [line.strip() for line in infile if line.strip()]
                else:
                    raise FileNotFoundError(f"No file found at {url}")
                url_full_text = self._delete_lines_not_containing_chars(url_full_text)
                url_full_text = self._normalize_spaces(url_full_text)
                all_text += "\n".join(url_full_text) + "\n\n"
                all_url_json.append({"url": url, "content": url_full_text})
            except Exception as e:
                self.logger.error(f"Error in {url}: {e}")
                continue
        if texttype == 'txt': return all_text, None
        elif texttype == 'json': return all_url_json, None




if __name__ == '__main__':



    logger = get_logger("TextExtraction")
    ext = Extractor(logger)

    base_read_dir = "exp/read"
    base_write_dir = "exp/write"
    langs = os.listdir(base_read_dir)
    texttype = 'txt'
    # file_types = ['.html']#, '.pdf', '.txt', '.xml', '.docx']
    # file_types = ['.pdf']#, , '.txt', '.xml', '.docx']
    # file_types = ['.xml']#, , '.txt', , '.docx']
    # file_types = ['.docx']#, , '.txt', , ]
    file_types = ['.pdf']#, , , , ]
    for i,lang in enumerate(langs):
        lang_dir = pjoin(base_read_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        for file_type in file_types:
            cp(f"Extracting {file_type} files from {i+1} {lang} ...")
            files = os.listdir(lang_dir)
            files = [file for file in files if file.endswith(file_type)]
            urls = [pjoin(lang_dir, file) for file in files]
            text, encoding_set = ext.extract_text(file_type, urls, texttype=texttype)
            outfile_bin = pjoin(base_write_dir, lang, lang + file_type + '.bin')
            outfile_txt = pjoin(base_write_dir, lang, lang + file_type + '.txt')
            outfile_json = pjoin(base_write_dir, lang, lang + file_type + '.json')
            os.makedirs(os.path.dirname(outfile_bin), exist_ok=True)
            ext.save_to_binary_file(text, outfile_bin)
            if texttype == 'txt': ext.save_to_txt_file(text, outfile_txt)
            elif texttype == 'json': ext.save_to_json_file(text, outfile_json)
            if encoding_set:
                cp(f'Encoding set for {file_type}: {encoding_set}', 'red')
                logger.info(f'Encoding set for {file_type}: {encoding_set}')
