"""
File: text_similarity_finder
Author: XF Zhang
Last updated on: 2024 Sep 11

text_similarity_finder module provides `TextSimilarity` classes that compute text similarity 
using methods from models including:
    - BERT (Bidirectional Encoder Representations from Transformers) and 
    - SBERT (Sentence-BERT).

    BERT's architecture:
    - bert-base-uncased: 12 layers (also called transformer blocks).
    - bert-large-uncased: 24 layers.

"""

# Dependencies

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from transformers import (
    BertTokenizer,
    BertModel,
    AutoModelForSequenceClassification,  # NLI model
    AutoTokenizer,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
import spacy

# Load spacy
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)  # spaCy stopwords


def convert_dict_to_array(similarity_dict):
    """
    Convert a dictionary of similarity scores to a list (array).

    Args:
        similarity_dict (dict): A dictionary containing similarity scores.

    Returns:
        list: A list of similarity scores.
    """
    return list(similarity_dict.values())


# Class for symmetrical similarity related metrics (text comparison order doesn't matter)
class TextSimilarity:
    """
    TextSimilarity class computes text similarities using the following methods:

    1. **Self-Attention Similarity**:
    - Computes the cosine similarity between the self-attention matrices of two texts.
    - Self-attention matrices capture the relationship between each word and every other word
    in the input text, which is crucial for understanding contextual dependencies.
    - When we refer to the self-attention of an entire text window (i.e., the full input sequence)
    in transformer models, we are referring to he last layer of the model.

    This method is sensitive to changes in word order and syntax but might not capture
    nuanced differences in semantic meaning as effectively as other methods.

    2. **Layer-Wise-Attention Similarity**:
    - Computes the cosine similarity between the self-attention matrices of two texts
        for EACH LAYER of a transformer model (e.g., BERT).
    - Averages these similarities to produce a single aggregate score.

    This method captures both syntactic and semantic similarities by analyzing attention patterns
    across all layers:
        - Lower layers focus on local relationships (e.g., word dependencies),
        - higher layers capture more abstract, global relationships (e.g., topic coherence).
        - Averaging these similarities provides a comprehensive measure of how similarly two texts are
    processed by the model -> useful for tasks like paraphrase detection and semantic similarity.

    3. **Hidden State Similarity**:
    - Computes the cosine similarity between the hidden states (representations) of two texts in BERT.
    - Hidden states from the last layer of BERT capture contextualized word embeddings.

    This method captures semantic relationships but can be influenced by subtle changes in wording.
    It is generally more effective than self-attention for capturing meaning but still relies on
    token-level similarities.

    4. **[CLS] Token Embedding Similarity** (CLS stands for classification):
    - Computes the cosine similarity between the [CLS] token embeddings of two texts.
    - The [CLS] token is a special token added at the beginning of every input sequence, and its
    corresponding hidden state in the final layer is used as a pooled representation
    of the entire sequence.

    This method is commonly used for classification tasks and provides a high-level summary of
    the input's meaning, making it effective for sentence-level similarity. However, it might miss
    finer nuances if the sentences are complex.

    5. **SBERT Sentence Embedding Similarity**:
    - Computes the cosine similarity between the sentence embeddings generated by SBERT (Sentence-BERT).
    - SBERT is fine-tuned specifically for generating semantically meaningful sentence embeddings.
    It uses a modified BERT architecture with pooling layers and is highly effective for capturing
    semantic similarities between sentences.

    This method tends to outperform the standard BERT-based methods when it comes to
    natural language understanding tasks that require semantic matching, paraphrase identification,
    or sentence clustering.

    6. **Semantic Textual Similarity (STS)**:
    - STS measures the degree to which two pieces of text (sentences, phrases, or paragraphs)
    express the same meaning or ideas.
    - The STS score ranges from 0 (completely dissimilar) to 1 (completely similar) &
    provides a nuanced understanding of text similarity by incorporating deep semantic insights.

    This method effectively captures complex semantic relationships such as
    negation, emphasis, paraphrasing, synonymy, and antonymy, making it useful for
    applications that require a deeper understanding of meaning.

    Each method provides unique insights into the texts' structure, syntax, and meaning,
    making the 'TextSimilarity' class a comprehensive tool for exploring text similarities
    using transformer-based models.

    However, the methods in the class are more suitable for comparing two texts with symmetrical
    relationships, and are not well adapted for asymmetrically related texts (i.e., how alike A is to B,
    but not B to A).

    Usage:
        To use the class,
        - create an instance of the `TextSimilarity` class &
        - call its methods with appropriate text inputs.

    Note: this class needs PADDING b/c we are performing comparisons between attention matrices
    and hidden states produced by a transformer-based model (like BERT) across entire sequences of
    tokens in a symmetrical manner.
    """

    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        sbert_model_name="all-MiniLM-L6-v2",
        sts_model_name="stsb-roberta-base",
    ):
        """
        Initialize models and tokenizers for BERT, SBERT, and STS.
        """
        # Load BERT models and tokenizer
        self.bert_model_attention = BertModel.from_pretrained(
            bert_model_name, output_attentions=True, attn_implementation="eager"
        )  # attn_implementation="eager" ->
        # the code is future-proof for Transformers v5.0.0 & beyond (the default PyTorch
        # implementation will no longer automatically fall back to the manual implementation.)
        self.bert_model_hidden_states = BertModel.from_pretrained(
            bert_model_name, output_hidden_states=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Load SBERT model
        self.sbert_model = SentenceTransformer(sbert_model_name)

        # Load STS model
        self.sts_model = SentenceTransformer(sts_model_name)

    def get_attention(self, input_text, context=None):
        """
        Get the attention scores for a given text using BERT.

        Attention scores provide insight into which words in a sentence are focusing on which other words.
        This method is useful for understanding how different words influence each other in a given text.
        """

        # Prepare text with context if provided
        text_wt_context = (
            [f"Context: {context}. Content: {input_text}"] if context else [input_text]
        )

        # Tokenize the input and compute attention scores
        input = self.tokenizer(
            text_wt_context, return_tensors="pt", padding=True, truncation=True
        )
        output = self.bert_model_attention(**input)

        # Extract attention scores from the output
        attentions = output.attentions  # A list of tensors for each layer

        return attentions

    def pad_to_match(self, tensor1, tensor2):
        """
        Pad two tensors to have the same dimensions.

        This is useful when comparing tensors of different sizes, such as attention matrices
        or hidden states. Padding ensures that the cosine similarity computation is performed
        on tensors of the same shape.
        """
        # Ensure inputs are tensors
        if isinstance(tensor1, tuple):
            tensor1 = tensor1[0]  # Convert tuple to tensor if necessary
        if isinstance(tensor2, tuple):
            tensor2 = tensor2[0]  # Convert tuple to tensor if necessary

        # Determine the max size in both dimensions (seq_len)
        max_size_0 = max(
            tensor1.size(0), tensor2.size(0)
        )  # Number of rows (sequence length)
        max_size_1 = max(
            tensor1.size(1), tensor2.size(1)
        )  # Number of columns (sequence length)

        # Pad tensor1 if needed
        pad_tensor1 = (0, max_size_1 - tensor1.size(1), 0, max_size_0 - tensor1.size(0))
        tensor1 = F.pad(tensor1, pad_tensor1, value=0)

        # Pad tensor2 if needed
        pad_tensor2 = (0, max_size_1 - tensor2.size(1), 0, max_size_0 - tensor2.size(0))
        tensor2 = F.pad(tensor2, pad_tensor2, value=0)

        return tensor1, tensor2

    def layer_wise_attention_similarity(self, text1, text2, context=None):
        """
        Compute the average cosine similarity of attention matrices across all layers
        of BERT for two texts.

        This method captures both syntactic and semantic similarities by analyzing
        attention patterns at different layers of the model.
        """
        # Get attentions for both texts
        attentions1 = self.get_attention(text1, context)
        attentions2 = self.get_attention(text2, context)

        # Initialize list to store cosine similarities for each layer
        layer_similarities = []

        # Iterate over layers and compute cosine similarity for each
        for layer_attention1, layer_attention2 in zip(attentions1, attentions2):
            # Mean over heads for each layer
            # (batch_size, num_heads, seq_len, seq_len)
            # -> (batch_size, seq_len, seq_len)
            attention1_mean = layer_attention1.mean(dim=1).squeeze(0)
            attention2_mean = layer_attention2.mean(dim=1).squeeze(0)

            # Pad attention matrices to have the same size
            attention1_mean, attention2_mean = self.pad_to_match(
                attention1_mean, attention2_mean
            )

            # Flatten attention matrices to compare
            attention1_flat = attention1_mean.view(-1)  # Flatten to a 1D tensor
            attention2_flat = attention2_mean.view(-1)  # Flatten to a 1D tensor

            # Compute cosine similarity for the current layer
            cosine_sim = torch.nn.functional.cosine_similarity(
                attention1_flat, attention2_flat, dim=0
            )
            layer_similarities.append(cosine_sim.item())

        # Compute the average similarity across all layers
        average_similarity = sum(layer_similarities) / len(layer_similarities)
        return average_similarity

    def self_attention_similarity(self, text1, text2, context=None):
        """
        Compute the cosine similarity between the attention matrices of two texts.

        This method compares the attention patterns between two sentences, which can reveal syntactic
        similarities but may not fully capture semantic differences.
        """
        # Get attentions for both texts
        attentions1 = self.get_attention(text1, context)
        attentions2 = self.get_attention(text2, context)

        # Extract the last layer's attention for both texts
        attention1 = (
            attentions1[-1].mean(dim=1).squeeze(0)
        )  # Mean over heads, shape (seq_len, seq_len)
        attention2 = attentions2[-1].mean(dim=1).squeeze(0)

        # Pad attention matrices to have the same size
        attention1, attention2 = self.pad_to_match(attention1, attention2)

        # Flatten attention matrices to compare
        attention1_flat = attention1.view(-1)  # Flatten to a 1D tensor
        attention2_flat = attention2.view(-1)  # Flatten to a 1D tensor

        # Ensure both tensors have the same size after padding and flattening
        assert (
            attention1_flat.size() == attention2_flat.size()
        ), "Padded tensors must have the same size."

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            attention1_flat, attention2_flat, dim=0
        )

        return cosine_sim

    def get_hidden_states(self, input_text, context=None):
        """
        Get the hidden states for a given text using BERT.
        """
        # Prepare text with context if provided
        text_wt_context = (
            [f"Context: {context}. Content: {input_text}"] if context else [input_text]
        )

        # Tokenize the input and compute hidden states
        inputs = self.tokenizer(
            text_wt_context, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.bert_model_hidden_states(**inputs)

        # Extract hidden states from the output (last hidden state)
        hidden_states = (
            outputs.last_hidden_state
        )  # Shape: (batch_size, seq_len, hidden_size)

        return hidden_states.squeeze(0)  # Remove batch dimension if batch_size = 1

    def self_hidden_state_similarity(self, text1, text2, context=None):
        """
        Compute the cosine similarity between the hidden states of two texts.
        """
        # Get hidden states for both texts
        hidden1 = self.get_hidden_states(text1, context)
        hidden2 = self.get_hidden_states(text2, context)

        # Pad hidden states to have the same size
        hidden1, hidden2 = self.pad_to_match(hidden1, hidden2)

        # Flatten hidden states to compare
        hidden1_flat = hidden1.view(-1)  # Flatten to a 1D tensor
        hidden2_flat = hidden2.view(-1)  # Flatten to a 1D tensor

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            hidden1_flat, hidden2_flat, dim=0
        )
        return cosine_sim

    def get_cls_embedding(self, input_text, context=None):
        """
        The [CLS] embedding is the embedding of the [CLS] token from the last hidden state of the BERT model.
        The [CLS] token is a special token added at the beginning of every input sequence,
        and its corresponding hidden state in the final layer is often used as a pooled representation of
        the entire sequence for classification and other sequence-level tasks (the final state.)

        IT DOES NOT NEED PADDING BECAUSE IT'S THE LAST LAYER ONLY!

        CLS stands for classification.
        """
        # Prepare text with context if provided
        text_wt_context = (
            [f"Context: {context}. Content: {input_text}"] if context else [input_text]
        )

        # Tokenize the input and compute hidden states
        inputs = self.tokenizer(
            text_wt_context, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.bert_model_hidden_states(**inputs)

        # Extract the [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[
            :, 0, :
        ]  # Shape: (batch_size, hidden_size)

        return cls_embedding.squeeze(0)  # Remove batch dimension if batch_size = 1

    def cls_embedding_similarity(self, text1, text2, context=None):
        """
        Compute the cosine similarity between the [CLS] token embeddings of two texts.

        CLS stands for classification.
        """
        # Get [CLS] token embeddings for both texts
        cls1 = self.get_cls_embedding(text1, context)
        cls2 = self.get_cls_embedding(text2, context)

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(cls1, cls2, dim=0)

        return cosine_sim

    def get_sentence_embedding(self, text):
        """
        Get the sentence embedding for a given text using SBERT (Sentence-BERT).
        """
        # Compute the embedding
        embedding = self.sbert_model.encode(text, convert_to_tensor=True)
        return embedding

    def sbert_similarity(self, text1, text2):
        """
        Compute the cosine similarity between sentence embeddings using SBERT (Sentence-BERT).
        """
        # Get embeddings for both texts
        embedding1 = self.get_sentence_embedding(text1)
        embedding2 = self.get_sentence_embedding(text2)

        # Compute cosine similarity
        cosine_sim = util.cos_sim(embedding1, embedding2)
        return cosine_sim.item()

    def sts_similarity(self, text1, text2):
        """
        Compute the semantic textual similarity (STS) between two texts using
        a fine-tuned model.

        More sensitive to subtle differences in meaning, STS measures the degree to
        which two texts express the same meaning or ideas.
        """
        # Get embeddings for both texts
        embedding1 = self.sts_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.sts_model.encode(text2, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_sim = util.cos_sim(embedding1, embedding2)
        return cosine_sim.item()

    def all_metrics(self, text1, text2, context=None):
        """
        Compute all similarity scores for a given pair of texts using various methods.

        Returns:
            dict: A dictionary containing similarity scores computed using different methods.
        """
        similarities = {}

        # Compute Self-Attention Similarity
        similarities["self_attention_similarity"] = self.self_attention_similarity(
            text1, text2, context
        ).item()

        # Compute Layer-Wise Attention Similarity
        similarities["layer_wise_attention_similarity"] = (
            self.layer_wise_attention_similarity(text1, text2, context)
        )

        # Compute Hidden State Similarity
        similarities["self_hidden_state_similarity"] = (
            self.self_hidden_state_similarity(text1, text2, context).item()
        )

        # Compute [CLS] Token Embedding Similarity
        similarities["cls_embedding_similarity"] = self.cls_embedding_similarity(
            text1, text2, context
        ).item()

        # Compute SBERT Sentence Embedding Similarity
        similarities["sbert_similarity"] = self.sbert_similarity(text1, text2)

        # Compute Semantic Textual Similarity (STS)
        similarities["sts_similarity"] = self.sts_similarity(text1, text2)

        return similarities

    def print_tensor(self, tensor):
        """
        Print a tensor in a formatted way.
        """
        for row in tensor:
            print(" ".join(f"{x:.2f}" for x in row))


# Class for asymmetrical similarity related metrics (text comparison order DOES matter)
class AsymmetricTextSimilarity:
    """
    A class to compute text similarities using methods that capture asymmetric relationships
    between texts (i.e., A is like B but not necessarily B is like A).

    This class is specifically designed for use cases where the order of comparison matters,
    such as matching responsibilities (candidate) to requirements (reference) or other similar
    relationships.

    Methods Overview:

    1. **BERTScore Precision**:
       - Computes BERTScore precision, which measures the overlap between the candidate's tokens
         and the reference's tokens using contextual embeddings generated by BERT.
       - BERTScore focuses on precision for asymmetric relationships: how much of the content in
         the reference (requirement) is covered by the candidate (responsibility/experience).
       - It provides a nuanced understanding of similarity at the token level and considers
         synonyms and contextual meanings.
       - Higher precision means more words in the candidate match those in the reference.

    2. **Soft Similarity**:
       - Computes the cosine similarity between sentence embeddings generated by SBERT (Sentence-BERT).
       - SBERT fine-tunes BERT for semantic textual similarity, making it more effective in capturing
         subtle differences in meaning.
       - This method is useful for semantic coverage, capturing high-level meaning rather than
         exact word matches.
       - A higher score indicates closer semantic alignment between the candidate and reference.

    3. **Word Mover's Distance (WMD)**:
       - Computes the Word Mover's Distance between the candidate and reference texts.
       - WMD measures the "distance" the words in the candidate need to travel in the semantic space
         to match the words in the reference.
       - It is a more sensitive metric that considers the cost of transforming one text into another.
       - Lower WMD indicates higher similarity; the method is useful for comparing overall text structures.

    4. **NLI Entailment Score**:
       - Uses a Natural Language Inference (NLI) model to compute the entailment score between two texts.
       - The model outputs probabilities for entailment, contradiction, and neutrality.
       - This method is asymmetric: it measures how much the candidate (hypothesis) is entailed by the
         reference (premise), focusing on directional reasoning.
       - A higher entailment score means the candidate (responsibility/experience) logically follows from
         the reference (requirement).

         (Note: NLI entailment is commented out for now b/c the existing NLI model is not optimal
         - needs finetuning.)

    5. **DeBERTa Entailment Score**:
        - Employs a variant of the BERT model, DeBERTa (Decoding-enhanced BERT with Disentangled Attention),
        to compute the entailment score between two texts.
        - The model outputs a probability distribution over entailment, contradiction, and neutrality,
        indicating the degree of semantic entailment between the texts.
        - The method is asymmetric: it measures how much the candidate (hypothesis) is semantically entailed
        by the reference (premise), focusing on directional reasoning and contextual relationships.
        - A higher DeBERTa entailment score -> the candidate (responsibility/experience) is more likely to be
        semantically entailed by the reference (requirement).

        (Note: although it is common convention to use:
        - job requirements as premise
        - resps/exp in resumes as hypothesis,
        I am flipping them around b/c logically it it is logical to infer the general from the specifics.
        Resps -> specific evidence (premise) -> general skillsets -> requirments (hypothesis))

    6. **Jaccard Similarity**:
       - Computes the Jaccard similarity, a set-based measure of the intersection over union of the tokens
         in both texts.
       - Jaccard similarity is symmetric: the order of the texts does not matter.
       - This metric is useful for partial coverage and exact word matches, focusing less on semantic meaning
         and more on the overlap of unique tokens.
       - A higher Jaccard similarity indicates a higher proportion of shared tokens between the texts.

    Usage:
        To use this module:
        - Create an instance of the `AsymmetricTextSimilarity` class.
        - Call its methods with the appropriate candidate (responsibility) and reference (requirement) texts.
    """

    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        sbert_model_name="all-MiniLM-L6-v2",
        deberta_model_name="microsoft/deberta-large-mnli",
        # nli_model_name="roberta-large-mnli",
        # There is something wrong with this model; I am swapping it out with deberta model for now!
    ):
        """
        Initialize models and tokenizers for BERT, SBERT, and NLI.
        """
        # Load BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)

        # Load SBERT model for semantic similarity
        self.sbert_model = SentenceTransformer(sbert_model_name)

        ### Swapping out for now
        # Load NLI model for entailment detection
        # self.nli_model = AutoModelForSequenceClassification.from_pretrained(
        #     nli_model_name
        # )
        # self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        ###

        # Load DeBERTa model for entailment detection
        self.deberta_model = AutoModelForSequenceClassification.from_pretrained(
            deberta_model_name
        )
        self.deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)

    def add_context(self, text1, text2, context):
        """Add context to each text if provided."""
        if context:
            text1 = f"{context} {text1}"
            text2 = f"{context} {text2}"
        return text1, text2

    def bert_score_precision(self, candidate, reference, context=None):
        """
        Compute BERTScore precision between texts with an asymmetrical similarity relationship.

        Args:
        - candidate (str): The text representing the responsibility or experience.
        - reference (str): The text representing the requirement.
        - context (str): Optional context to be added to both texts.

        Returns:
        - float: The BERTScore precision score.
        """
        # If context is provided, append it to the candidate sentence and reference paragraph
        candidate, reference = self.add_context(candidate, reference, context)

        # Calculate BERTScore (precision, recall, f1) for candidate and reference paragraph
        P, R, F1 = score([candidate], [reference], lang="en", verbose=True)
        return P.mean().item()

    def deberta_entailment_score(self, premise, hypothesis, context=None):
        """
        Compute the entailment score between two texts using a DeBERTa model.

        Args:
        - premise (str): The text representing the responsibility or experience.
        - hypothesis (str): The text representing the requirement.

        - context (str): Optional context to be added to both texts.

        Returns:
        - float: The probability that the hypothesis is entailed by the premise.
        """
        if context:
            hypothesis, premise = self.add_context(hypothesis, premise, context)

        # Encode premise (requirement) and hypothesis (responsibility)
        inputs = self.deberta_tokenizer.encode_plus(
            premise, hypothesis, return_tensors="pt", truncation=True
        )

        # Get model predictions (logits) for the inputs
        logits = self.deberta_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).tolist()[0]
        entailment_score = probs[2]  # Probability for entailment
        return entailment_score

    # def nli_entailment_score(self, hypothesis, premise, context=None):
    #     """
    #     Compute the entailment score between two texts using a Natural Language Inference (NLI) model.

    #     Args:
    #     - hypothesis (str): The text representing the responsibility or experience.
    #     - premise (str): The text representing the requirement.
    #     - context (str): Optional context to be added to both texts.

    #     Returns:
    #     - float: The probability that the hypothesis is entailed by the premise.
    #     """
    #     if context:
    #         hypothesis, premise = self.add_context(hypothesis, premise, context)

    #     # Encode premise (requirement) and hypothesis (responsibility)
    #     inputs = self.nli_tokenizer.encode_plus(
    #         premise, hypothesis, return_tensors="pt", truncation=True
    #     )

    #     # Get model predictions (logits) for the inputs
    #     logits = self.nli_model(**inputs).logits
    #     probs = torch.softmax(logits, dim=1).tolist()[0]
    #     entailment_score = probs[2]  # Probability for entailment
    #     return entailment_score

    def jaccard_similarity(self, text1, text2, context=None):
        """
        Compute Jaccard Similarity between two texts.

        Args:
        - text1 (str): The first text.
        - text2 (str): The second text.
        - context (str): Optional context to be added to both texts.

        Returns:
        - float: The Jaccard similarity score.
        """
        if context:
            text1, text2 = self.add_context(text1, text2, context)

        # Tokenize using spaCy and compute similarity metric
        tokens1 = set(token.text.lower() for token in nlp(text1))
        tokens2 = set(token.text.lower() for token in nlp(text2))

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0

    def soft_similarity(self, candidate, reference, context=None):
        """
        Compute soft similarity between two texts using SBERT embeddings.

        Args:
        - candidate (str): The text representing the responsibility or experience.
        - reference (str): The text representing the requirement.
        - context (str): Optional context to be added to both texts.

        Returns:
        - float: The cosine similarity score.
        """
        if context:
            candidate, reference = self.add_context(candidate, reference, context)

        # Compute the embeddings
        embedding1 = self.sbert_model.encode(candidate, convert_to_tensor=True)
        embedding2 = self.sbert_model.encode(reference, convert_to_tensor=True)

        # Compute soft cosine similarity using SBERT
        cosine_sim = util.cos_sim(embedding1, embedding2)
        return cosine_sim.item()

    def word_movers_distance(self, candidate, reference, context=None):
        """
        Compute the Word Mover's Distance (WMD) between two texts.

        Args:
        - candidate (str): The text representing the responsibility or experience.
        - reference (str): The text representing the requirement.
        - context (str): Optional context to be added to both texts.

        Returns:
        - float: The Word Mover's Distance. A smaller distance indicates more similarity.
        """
        if context:
            candidate, reference = self.add_context(candidate, reference, context)

        # Compute Word Mover's Distance
        # Use spaCy's stopwords
        vectorizer = CountVectorizer(stop_words=spacy_stopwords).fit(
            [candidate, reference]
        )

        vector1 = vectorizer.transform([candidate]).toarray()
        vector2 = vectorizer.transform([reference]).toarray()

        # Calculate the Euclidean distance
        distance = np.linalg.norm(vector1 - vector2)
        return distance

    def short_text_similarity_metrics(self, candidate, reference, context=None):
        """
        Compute all similarity metrics for a given pair of shorter texts
        - suited for segment to segment comparisons.



        Args:
        - candidate (str): The text representing the responsibility or experience.
        - reference (str): The text representing the requirement.
        - context (str): Optional context to be added to both texts.

        Returns:
            dict: A dictionary containing all computed similarity scores.
        """
        similarities = {}
        similarities["bert_score_precision"] = self.bert_score_precision(
            candidate, reference, context
        )
        similarities["soft_similarity"] = self.soft_similarity(
            candidate, reference, context
        )
        similarities["word_movers_distance"] = self.word_movers_distance(
            candidate, reference, context
        )
        similarities["deberta_entailment_score"] = self.deberta_entailment_score(
            candidate, reference, context
        )

        return similarities

    def all_metrics(self, candidate, reference, context=None):
        """
        Compute all similarity metrics for a given pair of texts.

        Args:
        - candidate (str): The text representing the responsibility or experience.
        - reference (str): The text representing the requirement.
        - context (str): Optional context to be added to both texts.

        Returns:
            dict: A dictionary containing all computed similarity scores.
        """
        similarities = {}
        similarities["bert_score_precision"] = self.bert_score_precision(
            candidate, reference, context
        )
        similarities["soft_similarity"] = self.soft_similarity(
            candidate, reference, context
        )
        similarities["word_movers_distance"] = self.word_movers_distance(
            candidate, reference, context
        )
        # similarities["nli_entailment_score"] = self.nli_entailment_score(
        #     candidate, reference, context
        # )
        similarities["deberta_entailment_score"] = self.deberta_entailment_score(
            candidate, reference, context
        )

        similarities["jaccard_similarity"] = self.jaccard_similarity(
            candidate, reference, context
        )

        return similarities


def compute_bertscore_precision(
    candidate_sent, ref_paragraph, candidate_context=None, reference_context=None
):
    """
    Calculates the BERTScore Precision, which measures how well the tokens in the
    candidate sentence are represented by the tokens in the reference paragraph.

    This metric introduces an asymmetry in the comparison because it focuses solely
    on how well the candidate sentence aligns with the reference paragraph. In other words,
    it measures the proportion of the candidate sentence's meaning that is covered by the
    reference paragraph, without considering how well the paragraph aligns with the sentence.

    Args:
        - candidate_sent (str): The candidate sentence whose alignment with the reference
        paragraph is being measured.
        - ref_paragraph (str): The reference paragraph against which the candidate sentence
        is compared.
        - candidate_context (str, optional):
        Additional context to include before or after the candidate sentence.
        Defaults to None.
        - reference_context (str, optional):
        Additional context to include before or after the reference paragraph.
        Defaults to None.

    Returns:
        float: The BERTScore Precision score, ranging from 0 to 1, indicating
        the degree of alignment of the candidate sentence (with context) with
        the reference paragraph (with context).

        A higher score indicates better alignment.
    """

    # If context is provided, append it to the candidate sentence and reference paragraph
    if candidate_context:
        candidate_sent = f"{candidate_context} {candidate_sent}"
    if reference_context:
        ref_paragraph = f"{reference_context} {ref_paragraph}"

    # Calculate BERTScore (precision, recall, f1) for candidate and ref parag
    P, R, F1 = score([candidate_sent], [ref_paragraph], lang="en", verbose=True)

    # Return only P
    return P.mean().item()


def main():
    """
    Main function for testing the similarity methods.
    """

    # # Example context and texts
    # context = "John is required to be at the event because he is the main speaker."
    # text1 = "John is present at the event."
    # text2 = "John is absent at the event."

    # # Example from Meta AI
    # context = "Embedded software development, memory management, C++"
    # text1 = """
    # Relying solely on C++ atomics and concurrency features is sufficient for ensuring thread-safe access
    # to shared data and protecting critical sections in multi-threaded embedded systems, providing a portable
    # and efficient solution.
    # """
    # text2 = """
    # While C++ atomics and concurrency features provide a foundation for thread safety,
    # they are insufficient on their own for ensuring data integrity and consistency in multi-threaded
    # embedded systems, and must be supplemented with platform-specific optimizations and custom solutions to
    # address the unique challenges of resource-constrained environments.
    # """

    # Example context and texts from GPT-4o
    context = "Artificial Intelligence in Healthcare, data privacy, machine learning models, patient data security."
    text1 = """
    Implementing standard machine learning models in healthcare applications, along with widely adopted data privacy
    protocols such as encryption and access control, is generally sufficient for ensuring patient data security. These
    methods provide a reliable foundation for preventing unauthorized access and maintaining data confidentiality in most scenarios.
    """
    text2 = """
    While standard machine learning models and commonly used data privacy protocols like encryption and access control provide
    a baseline level of security in healthcare applications, they are not sufficient on their own. Specialized approaches, such as
    differential privacy and secure multi-party computation, are needed to address the unique challenges of protecting sensitive
    patient data in complex machine learning environments.
    """

    # Print context, text1, text2
    print(f"Context: {context}\nText 1: {text1}\nText 2: {text2}\n")
    # Create an instance of the TextSimilarity class
    text_similarity = TextSimilarity()

    # Compute similarity using self-attention
    cosine_sim = text_similarity.self_attention_similarity(
        text1, text2, context=context
    )
    print(f"Cosine Similarity between attentions: {cosine_sim.item():.4f}")

    # Compute similarity using layer-wise attention analysis
    layer_attention_similarity = text_similarity.layer_wise_attention_similarity(
        text1, text2, context=context
    )
    print(
        f"Layer-Wise Attention Similarity (average across layers): {layer_attention_similarity:.4f}"
    )

    # Compute similarity using hidden states
    cosine_sim = text_similarity.self_hidden_state_similarity(
        text1, text2, context=context
    )
    print(f"Cosine Similarity between hidden states: {cosine_sim.item():.4f}")

    # Compute similarity using CLS embedding
    cosine_sim = text_similarity.cls_embedding_similarity(text1, text2, context=context)
    print(f"Cosine Similarity between [CLS] embeddings: {cosine_sim.item():.4f}")

    # Compute similarity using SBERT
    cosine_sim = text_similarity.sbert_similarity(text1, text2)
    print(f"Cosine Similarity using SBERT: {cosine_sim:.4f}")

    # Compute similarity using STS
    sts_similarity = text_similarity.sts_similarity(text1, text2)
    print(f"Cosine Similarity using STS: {sts_similarity:.4f}")


if __name__ == "__main__":
    main()