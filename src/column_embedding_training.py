import pandas as pd
from gensim.models import Word2Vec
import itertools
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

### Function to generate customer-product lists
def generate_customer_product_lists(df):
    """
    Generate a list of products purchased by each customer based on the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame where rows represent customers and columns represent products.

    Returns:
    list: List of lists containing products purchased by each customer.
    """
    return df.apply(lambda row: [product for product in df.columns if row[product] != 0], axis=1).tolist()

### Function to train a Word2Vec model on customer-product data
def train_word2vec(customers_products, vector_size=300, window=8, min_count=10, sg=1, hs=0, negative=1, workers=64, epochs=3):
    """
    Train a Word2Vec model on customer-product data.

    Parameters:
    customers_products (list): List of customer-product transactions.
    vector_size (int): Dimension of the word vectors.
    window (int): Maximum distance between the current and predicted word.
    min_count (int): Minimum count of words to consider.
    sg (int): Skip-gram model if 1, else CBOW.
    hs (int): Hierarchical softmax if 1, else negative sampling.
    negative (int): Number of negative samples.
    workers (int): Number of worker threads.
    epochs (int): Number of training epochs.

    Returns:
    Word2Vec: Trained Word2Vec model.
    """
    model = Word2Vec(sentences=customers_products, vector_size=vector_size, window=window, min_count=min_count,
                      sg=sg, hs=hs, negative=negative, workers=workers)
    model.train(customers_products, total_examples=len(customers_products), epochs=epochs)
    return model

### Function to generate column embeddings matrix
def generate_column_embeddings(df, model):
    """
    Generate embeddings for products (columns) in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame where columns represent products.
    model (Word2Vec): Trained Word2Vec model.

    Returns:
    np.ndarray: Matrix of product embeddings.
    """
    all_embeddings = []
    for product in df.columns:
        if product in model.wv.index_to_key:
            embedding = model.wv[product]
            all_embeddings.append(embedding)
    return np.array(all_embeddings)

### Function to visualize embeddings using t-SNE
def visualize_embeddings_tsne(embeddings, product_names):
    """
    Visualize product embeddings using t-SNE.

    Parameters:
    embeddings (np.ndarray): Product embeddings matrix.
    product_names (list): List of product names corresponding to the embeddings.
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    for i, product in enumerate(product_names):
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.annotate(product, (x, y), fontsize=10, alpha=0.75)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Product Embeddings')
    plt.grid(True)
    plt.show()

### Function to compute and visualize cosine similarity heatmap
def visualize_cosine_similarity_heatmap(embeddings, threshold):
    """
    Compute and visualize a heatmap of cosine similarities between product embeddings.

    Parameters:
    embeddings (np.ndarray): Product embeddings matrix.
    """
    cosine_sim_col = cosine_similarity(embeddings, embeddings)
    cosine_sim_col[cosine_sim_col < threshold] = 0

    col_df = pd.DataFrame(cosine_sim_col)
    plt.figure(figsize=(12, 8))
    sns.heatmap(col_df, cmap='viridis', annot=False)
    plt.title('Heatmap of the Column Embedding Similarities')
    plt.xlabel('Products')
    plt.ylabel('Products')
    plt.show()