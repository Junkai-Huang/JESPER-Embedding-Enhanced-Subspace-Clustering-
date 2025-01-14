import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils.convert import from_networkx
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity

### Function to compute joint embedding space
def compute_joint_embedding_space(customer_embeddings, column_embeddings_matrix, df, row_threshold, col_threshold):
    """
    Compute the joint embedding space using customer and column embeddings with cosine similarity.

    Parameters:
    customer_embeddings (np.ndarray): Embeddings for customers.
    column_embeddings_matrix (np.ndarray): Embeddings for columns (products).
    df (pd.DataFrame): Original data matrix.
    row_threshold (float): Threshold for row cosine similarity (default: 0.2).
    col_threshold (float): Threshold for column cosine similarity (default: 0.55).

    Returns:
    np.ndarray: Fuzzy joint embedding space matrix.
    """
    # Compute cosine similarity for rows (customers) and apply threshold
    cosine_sim_row = cosine_similarity(customer_embeddings, customer_embeddings)
    cosine_sim_row[cosine_sim_row < row_threshold] = 0

    # Compute cosine similarity for columns (products) and apply threshold
    cosine_sim_col = cosine_similarity(column_embeddings_matrix, column_embeddings_matrix)
    cosine_sim_col[cosine_sim_col < col_threshold] = 0

    # Compute the fuzzy joint embedding space
    fuzzy_matrix = cosine_sim_row @ df.values @ cosine_sim_col

    return fuzzy_matrix