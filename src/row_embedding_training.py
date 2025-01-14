import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils.convert import from_networkx
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

### Function to build a customer-product graph
def build_customer_product_graph(df):
    """
    Build a bipartite customer-product graph from a DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with rows as customers and columns as products.

    Returns:
    nx.Graph: Customer-product graph.
    """
    df_numpy = df.to_numpy()
    customer_product_graph = nx.Graph()

    for i in range(df_numpy.shape[0]):
        for j in range(df_numpy.shape[1]):
            if df_numpy[i][j] != 0:
                customer_product_graph.add_edge(i, j + df_numpy.shape[0], weight=df_numpy[i][j])

    print(f"Number of nodes in the graph: {customer_product_graph.number_of_nodes()}")
    print(f"Number of edges in the graph: {customer_product_graph.number_of_edges()}")
    return customer_product_graph

### Function to create edge index and weights from a graph
def create_edge_data(graph):
    """
    Create edge index and weight tensors from a graph.

    Parameters:
    graph (nx.Graph): Input graph.

    Returns:
    Data: PyTorch Geometric Data object containing edge index and weights.
    """
    edge_index = []
    weight = []

    for edge in graph.edges(data=True):
        u, v, w = edge
        w = w['weight']
        if u < v:
            edge_index.append([u, v])
        else:
            edge_index.append([v, u])
        weight.append(w)

    for edge in graph.edges(data=True):
        u, v, w = edge
        w = w['weight']
        if u > v:
            edge_index.append([u, v])
        else:
            edge_index.append([v, u])
        weight.append(w)

    edge_index = torch.tensor(np.array(edge_index, dtype=int).T)
    weight = torch.tensor(np.array(weight, dtype=int))

    data = Data(edge_index=edge_index, weight=weight, num_nodes=graph.number_of_nodes())
    return data

### Function to train Node2Vec model and extract embeddings
def train_node2vec(data, num_customers, num_epochs=8, embedding_dim=230, walk_length=120, 
                   context_size=10, walks_per_node=25, num_negative_samples=2, learning_rate=0.008):
    """
    Train a Node2Vec model and extract embeddings for customers and products.

    Parameters:
    data (Data): PyTorch Geometric Data object containing edge index and weights.
    num_customers (int): Number of customer nodes.
    num_epochs (int): Number of training epochs.
    embedding_dim (int): Dimension of the embeddings.
    walk_length (int): Length of random walks.
    context_size (int): Size of the context window.
    walks_per_node (int): Number of walks per node.
    num_negative_samples (int): Number of negative samples per walk.
    learning_rate (float): Learning rate for optimization.

    Returns:
    tuple: Customer embeddings and product embeddings as numpy arrays.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                     context_size=context_size, walks_per_node=walks_per_node,
                     num_negative_samples=num_negative_samples, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=64, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # List to store the loss values for each epoch
    loss_values = []

    with tqdm(total=num_epochs, desc='Training Progress') as pbar:
        for epoch in range(1, num_epochs + 1):
            loss = train()
            loss_values.append(loss)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    # Extract embeddings for customers and products
    customer_embeddings = model(torch.arange(num_customers, device=device)).detach().cpu().numpy()
    num_products = data.num_nodes - num_customers
    product_embeddings = model(torch.arange(num_customers, num_customers + num_products, device=device)).detach().cpu().numpy()

    # Plot the loss values to visualize convergence
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Convergence over Epochs')
    plt.grid(True)
    plt.show()

    return customer_embeddings, product_embeddings
