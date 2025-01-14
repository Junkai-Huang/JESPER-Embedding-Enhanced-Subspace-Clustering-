from column_embedding_training import *
from row_embedding_training import *
from biclustering_on_joint_embedding_space import *
from compute_joint_embedding_space import *
from dataframe_generation import *

def main():
    # Step 1: Generate synthetic dataframe
    print("Generating synthetic dataframe...")
    df_noisy, df_groundtruth = generate_dataframe()

    # Step 2: Train column embeddings
    print("Training column embeddings...")
    customers_products = generate_customer_product_lists(df_noisy)
    column_embedding_model = train_word2vec(customers_products, vector_size=300, window=8, min_count=10, sg=1, hs=0, negative=1, workers=64, epochs=3)
    column_embeddings_matrix = generate_column_embeddings(df_noisy, column_embedding_model)

    # Visualize column embeddings
    print("Visualizing column embeddings...")
    product_names = df_noisy.columns.tolist()
    visualize_embeddings_tsne(column_embeddings_matrix, product_names)
    visualize_cosine_similarity_heatmap(column_embeddings_matrix, threshold=0.55)

    # Step 3: Train row embeddings
    print("Training row embeddings...")
    customer_product_graph = build_customer_product_graph(df_noisy)
    edge_data = create_edge_data(customer_product_graph)
    num_customers = len(df_noisy)
    row_embeddings, product_embeddings = train_node2vec(edge_data, num_customers)

    # Step 4: Compute joint embedding space
    print("Computing joint embedding space...")
    joint_embedding_space = compute_joint_embedding_space(
        row_embeddings, column_embeddings_matrix, df_noisy, row_threshold=0.2, col_threshold=0.55
    )

    # Step 5: Select biclustering algorithm
    print("Select the biclustering algorithm:")
    print("1: LAS")
    print("2: Plaid")
    print("3: CCA")
    print("4: Spectral Biclustering")
    print("5: CoclustMod")
    print("6: CoclustInfo")
    choice = int(input("Enter the number corresponding to your choice: "))

    # Step 6: Execute the selected biclustering algorithm
    if choice == 1:
        # LAS Parameters
        num_biclusters = int(input("Enter the number of biclusters: "))
        score_threshold = float(input("Enter the score threshold: "))
        randomized_searches = int(input("Enter the number of randomized searches: "))
        tol = float(input("Enter the tolerance: "))
        run_las(joint_embedding_space, num_biclusters=num_biclusters, score_threshold=score_threshold, 
                randomized_searches=randomized_searches, tol=tol)
    elif choice == 2:
        # Plaid Parameters
        num_biclusters = int(input("Enter the number of biclusters: "))
        row_pruning_threshold = float(input("Enter the row pruning threshold: "))
        col_pruning_threshold = float(input("Enter the column pruning threshold: "))
        back_fitting_steps = int(input("Enter the number of back-fitting steps: "))
        initialization_iterations = int(input("Enter the number of initialization iterations: "))
        iterations_per_layer = int(input("Enter the number of iterations per layer: "))
        significance_tests = int(input("Enter the number of significance tests: "))
        run_plaid(joint_embedding_space, num_biclusters=num_biclusters, row_pruning_threshold=row_pruning_threshold, 
                  col_pruning_threshold=col_pruning_threshold, back_fitting_steps=back_fitting_steps,
                  initialization_iterations=initialization_iterations, iterations_per_layer=iterations_per_layer,
                  significance_tests=significance_tests)
    elif choice == 3:
        # CCA Parameters
        num_biclusters = int(input("Enter the number of biclusters: "))
        msr_threshold = float(input("Enter the MSR threshold: "))
        multiple_node_deletion_threshold = float(input("Enter the multiple node deletion threshold: "))
        run_cca(joint_embedding_space, num_biclusters=num_biclusters, msr_threshold=msr_threshold, 
                multiple_node_deletion_threshold=multiple_node_deletion_threshold)
    elif choice == 4:
        # Spectral Biclustering Parameters
        n_clusters = tuple(map(int, input("Enter the number of row and column clusters (e.g., '7,7'): ").split(',')))
        method = input("Enter the method ('log' or 'bistochastic'): ")
        random_state = int(input("Enter the random state: "))
        run_spectral_biclustering(joint_embedding_space, n_clusters=n_clusters, method=method, random_state=random_state)
    elif choice == 5:
        # CoclustMod Parameters
        n_clusters = int(input("Enter the number of clusters: "))
        run_coclustmod(joint_embedding_space, n_clusters=n_clusters)
    elif choice == 6:
        # CoclustInfo Parameters
        n_row_clusters = int(input("Enter the number of row clusters: "))
        n_col_clusters = int(input("Enter the number of column clusters: "))
        run_coclustinfo(joint_embedding_space, n_row_clusters=n_row_clusters, n_col_clusters=n_col_clusters)
    else:
        print("Invalid choice. Please select a valid algorithm.")

if __name__ == "__main__":
    main()
