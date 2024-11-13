"""compute_similarity.py"""

import logging
import logging_config
from pathlib import Path
import json
import numpy as np
from transformers import BertModel, BertTokenizer, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster

# Setup logging
logger = logging.getLogger(__name__)


def compute_similarity_and_cluster(
    thoughts, main_topic, model_name="bert-base-uncased", top_n=3
):
    """
    Compute self-attention based similarity and cluster distinct sub-thoughts.

    Args:
        thoughts (dict): Dictionary containing sub-thoughts generated from LLM.
        main_topic (str): Main topic to be include with each sub-thought to set context.
        model_name (str): Transformer model name for computing self-attention. Default is "bert-base-uncased".
        top_n (int): Number of top distinct sub-thoughts to select. Default is 3.

    Returns:
        tuple: Top N distinct sub-thoughts based on clustering and the cosine similarity matrix.
    """
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained(model_name)

    # Preprocess sub-thoughts with context
    sub_thoughts = [thought["thought_content"] for thought in thoughts["thoughts"]]
    topics_with_context = [
        f"Context: {main_topic}. Topic: {sub_thought}" for sub_thought in sub_thoughts
    ]

    # Tokenize and compute attention scores
    # (Use AutoTokenizer and attention_mask b/c the clean_up_tokenization_spaces parameter
    # will change its default value from False to True in transformers version 4.45.)
    output = tokenizer(
        topics_with_context,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = output["input_ids"]  # each input_id is a tensor
    # tokenizer returns a dictionary with several keys: input_ids, attention_mask, and token_type_ids.
    # We're only interested in the input_ids -> access the dictionary with ["input_ids"] only.

    # Attention mask (optional for later use)
    attention_mask = output[
        "attention_mask"
    ]  # attention_mask "masks the future" -> only pay attention to the past!

    # Get attention info: using a short-cut, a quirk specific to the BERT model
    # [:, 0, :] slice is not exactly the attention score; it's a specific part of the last hidden state
    # that contains the attention info; the slice extracts the first token's hidden state for each sequence in the batch
    # (the 1st tok is the [CLS] token -> its hidden state has the aggregate information from the entire sequence,
    # including the attention weights.
    attention_scores = model(input_ids).last_hidden_state[:, 0, :]
    attention_scores = (
        attention_scores.detach().numpy()
    )  # turn tensor (pt) into numpy.array

    # Compute dis-similarity matrix
    cosine_sim_matrix = cosine_similarity(attention_scores)
    dis_sim_matrix = 1 - cosine_sim_matrix

    # Cluster and select top 3 distinct sub-thoughts
    clusters = linkage(dis_sim_matrix, method="ward")
    cluster_ids = fcluster(clusters, top_n, criterion="maxclust")
    top_n_distinct = [
        (
            thoughts["thoughts"][i]["thought_name"],
            thoughts["thoughts"][i]["thought_content"],
        )
        for i in np.argsort(cluster_ids)[:top_n]
    ]
    return top_n_distinct, cosine_sim_matrix
