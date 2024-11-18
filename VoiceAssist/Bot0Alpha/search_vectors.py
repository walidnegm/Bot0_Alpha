# search_vectors.py
# pip install sentence-transformers

import numpy as np
import pickle
from common_modules import *
#from llama_prompt_call import llama_prompt_call

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Configure logging

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
TAG = "vector_search"
FILE_DOCS_AND_EMBEDDINGS = "docs_and_embeddings.pkl"
DOCS_FILE = "docs.pkl"


class vector_search:
    def __init__(self):
        self.docs = None
        self.docs_embeddings = None
        self.model = None

    def load_embeddings(self):

        # 1. Specify preferred dimensions

        dimensions = 512
        # 2. Load model
        self.model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)
        # Load the docs and embeddings from the file

        with open(FILE_DOCS_AND_EMBEDDINGS, "rb") as f:
            data = pickle.load(f)
            self.docs_embeddings = data


        with open(DOCS_FILE, "rb") as f:
            data = pickle.load(f)
            self.docs = data

    def search_vector_store(self, query):
        # The prompt used for query retrieval tasks
        query_prompt = "Represent this sentence for searching relevant passages:"

        # 2. Encode
        query_embedding = self.model.encode(query_prompt + query)

        # Calculate cosine similarities
        similarities = cos_sim(query_embedding, self.docs_embeddings)
        similarities = similarities.squeeze()
        top_5_similarities = np.argsort(-similarities)[:5]
        print("Top 5 similarities:", similarities[top_5_similarities])

        top_5_docs = [self.docs[i] for i in top_5_similarities]
        print("Top 5 docs:", top_5_docs)
        return top_5_docs