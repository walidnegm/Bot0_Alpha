# Install necessary packages

import PyPDF2
import re
import logging
import pickle  # For saving the text mapping

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim



# Configure logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


TAG = "store_vectors"
FILE_DOCS_AND_EMBEDDINGS = "docs_and_embeddings.pkl"
DOCS_FILE = "docs.pkl"

# Define a function to preprocess text to remove newlines and tabs

def preprocess_text_1(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    return text



# Define a function to preprocess text to split into sentences

def preprocess_text_2(text):
    text = preprocess_text_1(text)
    sentences = re.split(r'(?<=[.!?])+', text)
    return sentences

# Define a function to extract text from PDF

def extract_text_from_pdf(pdf_path):
    texts = ""
    reader = PyPDF2.PdfReader(pdf_path)
    for page in reader.pages:
        texts += page.extract_text()
    return texts

if __name__ == "__main__":
    file_path = "/root/backend/MY23_Prius_OM_Excerpt_for_Driving_Support_Systems_D4_ML_0208.pdf"  # Replace with your actual file path


    # 1. Specify preferred dimensions

    dimensions = 512


    # 2. Load model
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)


    # The prompt used for query retrieval tasks:
    query_prompt = "Represent this sentence for searching relevant passages: "

    query = "what does the engine indicator light mean"

    # 2. Encode
    query_embedding = model.encode(query_prompt + query)


    # Process PDF and create embeddings

    texts = extract_text_from_pdf(file_path)
    processed_text = preprocess_text_2(texts)
    logging.debug(f"{TAG} Processing PDF done")

    docs = processed_text


    # Encode the documents

    docs_embeddings = model.encode(docs)



    logging.debug(f"{TAG} PDF embeddings created")



    # Save the docs and embeddings to a file

    with open(FILE_DOCS_AND_EMBEDDINGS, 'wb') as f:
        pickle.dump(docs_embeddings, f)

    with open(DOCS_FILE, 'wb') as f:
        pickle.dump(docs, f)

    logging.debug(f"{TAG} Store PDF to pkl file completed")

