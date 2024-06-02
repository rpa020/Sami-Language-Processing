from flask import Flask, render_template, request, send_file
import os
from plagiarism import PlagiarismChecker
import argparse

parser = argparse.ArgumentParser(description='Arguments used in user interface for plagiarims checker')

parser.add_argument('model', type=str, help='Path to model')
parser.add_argument('tokenizer', type=str, help='Path to tokenizer')
parser.add_argument('database', type=str, help='Path to chromaDB')
parser.add_argument('--collection', type=str, default='test', help='Choose collection in database (test)')


app = Flask(__name__, '/static')

@app.route('/')
def home():
  return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():

  text = request.form.get('text_input')
  with open("app_tmp.txt", "w") as f:
    f.write(text)

  jdata = p.extract_data("app_tmp.txt", 'txt')
  res = p.plagiarism_checker(jdata)
  
  return render_template("res.html", result=res)

# change testdata to working dataset
@app.route(f'/download')
def download_file():
    file_path = request.args.get('path')

    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found: document is not added in the server", 404
    

if __name__ == '__main__':
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
    app.run(debug=True, host='0.0.0.0', port=8000)
  else:
    raise Exception(f"Collection does not exist. Only these collections exists: {p.list_collections()}")
