import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from biclustlib.algorithms import LargeAverageSubmatrices, Plaid, ChengChurchAlgorithm
from sklearn.cluster import SpectralBiclustering
from cclust_package.coclust.coclustering import CoclustMod, CoclustInfo
import matplotlib.pyplot as plt

### Utility Functions

def format_indices(indices):
    """
    Format the list of indices to show only the start and the end if the list is long.
    """
    if len(indices) <= 20:  # If the list is short, show all
        return indices
    return indices[:10] + ['...'] + indices[-10:]  # Show first 10, '...', and last 10

def format_biclusters(bicluster_list):
    """
    Format biclusters by sorting row and column indices.
    """
    formatted_biclusters = []
    for bicluster in bicluster_list:
        rows = sorted(bicluster['rows'])
        cols = sorted(bicluster['columns'])
        formatted_biclusters.append((list(rows), list(cols)))
    return formatted_biclusters

### LAS Algorithm Function

def run_las(df, num_biclusters=7, score_threshold=1.0, randomized_searches=1000, tol=1e-6):
    """
    Run the Large Average Submatrices (LAS) algorithm.
    """
    las = LargeAverageSubmatrices(
        num_biclusters=num_biclusters,
        score_threshold=score_threshold,
        randomized_searches=randomized_searches,
        transform=False,
        tol=tol
    )
    result = las.run(df.values)

    print("Identified Biclusters:")
    for idx, bicluster in enumerate(result.biclusters):
        formatted_rows = format_indices(sorted(bicluster.rows))
        formatted_columns = format_indices(sorted(bicluster.cols))
        print(f"Bicluster {idx + 1}")
        print("Rows:", formatted_rows)
        print("Columns:", formatted_columns)

### Plaid Algorithm Function

def run_plaid(df, num_biclusters=7, row_pruning_threshold=0.01, col_pruning_threshold=0.01, 
              back_fitting_steps=3, initialization_iterations=10, iterations_per_layer=10, significance_tests=1):
    """
    Run the Plaid algorithm.
    """
    plaid = Plaid(
        num_biclusters=num_biclusters,
        row_prunning_threshold=row_pruning_threshold,
        col_prunning_threshold=col_pruning_threshold,
        back_fitting_steps=back_fitting_steps,
        initialization_iterations=initialization_iterations,
        iterations_per_layer=iterations_per_layer,
        significance_tests=significance_tests,
        fit_background_layer=True
    )
    result = plaid.run(df.values)

    print("Identified Biclusters:")
    for idx, bicluster in enumerate(result.biclusters):
        formatted_rows = format_indices(sorted(bicluster.rows))
        formatted_columns = format_indices(sorted(bicluster.cols))
        print(f"Bicluster {idx + 1}")
        print(len(bicluster.rows), len(bicluster.cols))
        print("Rows:", formatted_rows)
        print("Columns:", formatted_columns)

### Cheng-Church Algorithm Function

def run_cca(df, num_biclusters=7, msr_threshold=1000, multiple_node_deletion_threshold=1.2):
    """
    Run the Cheng-Church Algorithm (CCA).
    """
    cca = ChengChurchAlgorithm(
        num_biclusters=num_biclusters,
        msr_threshold=msr_threshold,
        multiple_node_deletion_threshold=multiple_node_deletion_threshold
    )
    result = cca.run(df.values)

    print("Identified Biclusters:")
    for idx, bicluster in enumerate(result.biclusters):
        formatted_rows = format_indices(sorted(bicluster.rows))
        formatted_columns = format_indices(sorted(bicluster.cols))
        print(f"Bicluster {idx + 1}")
        print(len(bicluster.rows), len(bicluster.cols))
        print("Rows:", formatted_rows)
        print("Columns:", formatted_columns)

### Spectral Biclustering Function

def run_spectral_biclustering(df, n_clusters=(7, 7), method='log', random_state=0):
    """
    Run the Spectral Biclustering algorithm.
    """
    clustering = SpectralBiclustering(n_clusters=n_clusters, method=method, random_state=random_state)
    clustering.fit(df.values)

    bicluster_list = []
    for i in range(clustering.n_clusters[0]):
        rows = np.where(clustering.row_labels_ == i)[0]
        columns = np.where(clustering.column_labels_ == i)[0]
        bicluster_list.append({"rows": rows, "columns": columns})

    result = format_biclusters(bicluster_list)

    for idx, bicluster in enumerate(result):
        print(f"Bicluster {idx}: rows = {bicluster[0]}, columns = {bicluster[1]}")

### CoclustMod Function

def run_coclustmod(df, n_clusters=7):
    """
    Run the CoclustMod algorithm.
    """
    model = CoclustMod(n_clusters=n_clusters)
    model.fit(df.values)

    row_labels = model.row_labels_
    column_labels = model.column_labels_

    bicluster_list = []
    for label in np.unique(row_labels):
        rows = np.where(row_labels == label)[0]
        columns = np.where(column_labels == label)[0]
        bicluster_list.append({"rows": rows, "columns": columns})

    result = format_biclusters(bicluster_list)

    for idx, bicluster in enumerate(result):
        print(f"Bicluster {idx}: rows = {bicluster[0]}, columns = {bicluster[1]}")

### CoclustInfo Function

def run_coclustinfo(df, n_row_clusters=7, n_col_clusters=7):
    """
    Run the CoclustInfo algorithm.
    """
    model = CoclustInfo(n_row_clusters=n_row_clusters, n_col_clusters=n_col_clusters)
    model.fit(df.values)

    row_labels = model.row_labels_
    column_labels = model.column_labels_

    bicluster_list = []
    for label in np.unique(row_labels):
        rows = np.where(row_labels == label)[0]
        columns = np.where(column_labels == label)[0]
        bicluster_list.append({"rows": rows, "columns": columns})

    result = format_biclusters(bicluster_list)

    for idx, bicluster in enumerate(result):
        print(f"Bicluster {idx}: rows = {bicluster[0]}, columns = {bicluster[1]}")