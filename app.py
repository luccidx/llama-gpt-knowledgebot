from flask import Flask, request, jsonify
import os
os.environ['CURL_CA_BUNDLE'] = ''
import faiss
from sentence_transformers import SentenceTransformer
import gpt4all
import threading

import requests
# response = requests.request("POST", 'https://gpt4all.io/models/models.json', headers=headers, data=payload, verify=False)
# response = requests.get('https://gpt4all.io/models/models.json', verify=False)

app = Flask(__name__)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load corpus
data_dir = "./Porsche wiki"
corpus = []

for filename in os.listdir(data_dir):
    print(f"Loading file {filename}")
    with open(f"{data_dir}/{filename}", 'r', encoding='utf-8') as f:
        doc = f.readlines()
        corpus.append(doc)

# Build Embeddings
embeddings = model.encode(corpus)

# Build the FAISS index
d = embeddings.shape[1]
nlist = 40
newindex = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
newindex.train(embeddings)
newindex.add(embeddings)

# Load GPT-4 model
gptj = gpt4all.GPT4All("llama-2-7b-chat.ggmlv3.q4_0.bin")

@app.route("/")
def index():
    return open("index.html").read()

@app.route("/get-response")
def get_response():
    user_input = request.args.get("user_input")
    response = generate_response(user_input)
    return jsonify({"response": response})

def generate_response(user_input):
    xq = model.encode([user_input])
    k = 1
    D, I = newindex.search(xq, k)
    most_similar_document = corpus[I[0][0]]
    context = " ".join(most_similar_document)
    question = user_input
    input_text = f"<context>{context}<context><question>{question}<question>"
    # Split the input text into chunks
    max_tokens = 2040  # Adjust this value as needed
    answer = gptj.generate(input_text)
    return answer

def generate_response_threaded(user_input):
    response = None
    thread = threading.Thread(target=generate_response, args=(user_input,))
    thread.start()
    thread.join()  # Wait for the thread to finish
    return response

if __name__ == "__main__":
    app.run(debug=True)





