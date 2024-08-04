[![luccidx/llama-gpt-knowledgebot - GitHub](https://gh-card.dev/repos/luccidx/llama-gpt-knowledgebot.png?fullname=)](https://github.com/luccidx/llama-gpt-knowledgebot)

# Project: Text-Based Question Answering System with Context Retrieval

This project implements a question-answering system that leverages pre-trained models and semantic similarity search to provide informative responses based on a given context.

## Key Features

- **Context Retrieval**: Utilizes Sentence Transformers to generate embeddings for both user input and stored documents (corpus).
- **Semantic Similarity Search**: Employs FAISS (Facebook AI Similarity Search) to efficiently identify the most relevant documents to the user's question.
- **Context-Aware Response Generation**: Combines the retrieved context with the user's question and feeds it into a GPT-4 model for response generation.

## Breakdown of the Code

### Imports

- **Flask**: Web framework for building the API server.
- **Request**: Handling user input from the web interface.
- **Jsonify**: Converting responses to JSON format for efficient transmission.
- **OS**: Setting environment variables (currently unused).
- **Faiss**: Library for fast similarity search.
- **Sentence Transformers**: Pre-trained model for generating text embeddings.
- **GPT4all**: Client library for accessing the GPT-4 model.
- **Threading**: Enabling threaded response generation (commented out currently).
- **Requests**: Library for making HTTP requests (commented out currently).

### Flask Application Setup

```python
app = Flask(__name__)  # Creates a Flask application instance.
```
### Sentence Transformer Model Loading

To generate text embeddings for both user input and the corpus, we use a pre-trained Sentence Transformer model:

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
```
- **SentenceTransformer('all-MiniLM-L6-v2')**: This command loads the 'all-MiniLM-L6-v2' model, which is a lightweight, fast model suitable for generating embeddings for semantic textual similarity tasks.
- The model generates dense vector representations (embeddings) for text, which can be used for various tasks, including similarity search, clustering, and classification.

### Corpus Loading
- **data_dir**: Path to the directory containing the corpus documents (e.g., Porsche wiki articles).
- The code iterates through files in the data directory, reads their contents, and builds the corpus as a list of document lists.
Embedding Generation

### Embedding Generation

```python
embeddings = model.encode(corpus)  # Generates embeddings for each document in the corpus
```

### FAISS Index Construction

```python
d = len(embeddings[0])  # Dimensionality of the embedding vectors.
nlist = 10  # Number of neighbors to consider during search.
newindex = faiss.IndexFlatL2(d)  # Creates a FAISS index optimized for efficient similarity search.
newindex.train(embeddings)  # Trains the index on the generated embeddings.
newindex.add(embeddings)  # Adds the corpus embeddings to the index.
```

### GPT-4 Model Loading

```python
gptj = gpt4all.GPT4All("llama-2-7b-chat.ggmlv3.q4_0.bin")  # Loads the GPT-4 model for text generation.
```

### API Endpoints
- **/**: Serves the static HTML content (presumably the web interface).
- **/get-response**: Handles user input, retrieves the response using generate_response, and returns the generated answer in JSON format.


### Response Generation Function (generate_response)

```python
xq = model.encode([user_input])  # Generates an embedding for the user's input.
k = 1  # Specifies the number of nearest neighbors to retrieve.
D, I = newindex.search(xq, k)  # Searches the FAISS index for the nearest neighbor (most similar document) to the user input.
most_similar_document = corpus[I[0][0]]  # Extracts the most relevant document from the corpus based on the search results.
context = " ".join(most_similar_document)  # Concatenates the retrieved document content into a single string.
question = user_input  # Stores the user's input as the question.
input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"  # Prepares the input text for GPT-4, combining context and question with placeholders.
max_tokens = 100  # Sets a maximum token limit for GPT-4 input (adjustable).
answer = gptj.generate(input_text)  # Generates the response text using the GPT-4 model with the prepared context and question.
```

### Threaded Response Generation (Commented Out)

```python
def generate_response_threaded(user_input):
    response_thread = threading.Thread(target=generate_response, args=(user_input,))
    response_thread.start()
    response_thread.join()
```

### Application Execution

```python
if __name__ == "__main__":
    app.run(debug=True)  # Starts the Flask application in debug mode, allowing for automatic code reloading during development.
```
