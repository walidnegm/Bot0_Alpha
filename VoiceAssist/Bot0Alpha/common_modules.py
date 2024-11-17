from datetime import time
import re
#from pinecone.grpc import PineconeGRPC as Pinecone
#from pinecone import ServerlessSpec
import PyPDF2
import logging



# Configure the logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

TAG = "common_modules"



# Delete pinecone index

def delete_index(pc, index_name):

    pc.delete_index(index_name)



# Create pinecone index

def create_index(pc, index_name, is_delete_index=True):
    if is_delete_index:
        # delete old index
        pc.delete_index(pc, index_name)

    # List all existing indexes
    indexes_information = pc.list_indexes()
    logging.debug(f"{TAG} Indexes information: {indexes_information}")



    # Extracting index names
    index_names = [index['name'] for index in indexes_information]
    logging.debug(f"{TAG} Index names that are already available: {index_names}")


    # Check if the specific index exists
    if index_name in index_names:
        logging.debug(f"{TAG} Index exists")
    else:
        logging.debug(f"{TAG} Index does not exist. Creating index.")
        pc.create_index({
            "name": index_name,
            "dimension": 1536,
            "metric": "cosine",
            "spec": ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        })
        logging.debug(f"{TAG} Index created")



    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    logging.debug(f"{TAG} Index found and ready")



# Define a function to preprocess text to remove newlines and tabs

def preprocess_text_1(text):

    # Replace consecutive spaces, newlines and tabs

    text = re.sub(r'\s+', ' ', text)

    return text



# Define a function to preprocess text to split into sentences

def preprocess_text_2(text):
    text = preprocess_text_1(text)
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences



# Define a function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    texts = ""
    reader = PyPDF2.PdfReader(pdf_path)
    for page in reader.pages:
        texts += page.extract_text()
    return texts



# Define a function to create embeddings
def create_embeddings(client, texts, model):
    logging.debug(f"{TAG} Creating embeddings")
    logging.debug(f"{TAG} Length of texts: ", len(texts))
    embeddings_list = []

    try:
        res = client.embeddings.create(input=texts, model=model)
        embeddings_list = [item.embedding for item in res.data]

    except Exception as e:
        logging.error(f"{TAG} An error occurred: {e}")
    logging.debug(f"{TAG} Embedding created: ", len(embeddings_list))
    return embeddings_list