# Joint Embedding Space Biclustering Framework

This repository contains a framework for performing biclustering on a joint embedding space derived from row and column embeddings of a synthetic dataset. The pipeline includes generating data, training embeddings, computing the joint embedding space, and applying various biclustering algorithms.

## Project Structure

```
src/
├── biclustering_on_joint_embedding_space.py   # Implements biclustering algorithms
├── column_embedding_training.py              # Trains column embeddings using Word2Vec
├── compute_joint_embedding_space.py          # Computes joint embedding space
├── dataframe_generation.py                   # Generates synthetic data
├── main.py                                   # Main pipeline
├── row_embedding_training.py                 # Trains row embeddings using Node2Vec
README.md
requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Features

1. **Synthetic Data Generation**
   - Generates a synthetic customer-product dataset with bicluster structures and noise.
   - Located in `dataframe_generation.py`.

2. **Column Embedding Training**
   - Uses Word2Vec to train embeddings for product columns.
   - Visualizes embeddings using t-SNE and cosine similarity heatmaps.
   - Located in `column_embedding_training.py`.

3. **Row Embedding Training**
   - Uses Node2Vec to train embeddings for customer rows based on a bipartite graph.
   - Located in `row_embedding_training.py`.

4. **Joint Embedding Space Computation**
   - Combines row and column embeddings to compute a joint embedding space.
   - Located in `compute_joint_embedding_space.py`.

5. **Biclustering on Joint Embedding Space**
   - Provides various biclustering algorithms to analyze the joint embedding space, including:
     - LAS (Large Average Submatrices)
     - Plaid
     - Cheng-Church Algorithm (CCA)
     - Spectral Biclustering
     - CoclustMod
     - CoclustInfo
   - Located in `biclustering_on_joint_embedding_space.py`.

6. **Pipeline Execution**
   - A complete pipeline to execute the entire workflow.
   - Located in `main.py`.

## Usage

Run the main pipeline:
```bash
python src/main.py
```

The pipeline performs the following steps:
1. Generate synthetic data.
2. Train column embeddings using Word2Vec.
3. Train row embeddings using Node2Vec.
4. Compute the joint embedding space.
5. Apply a selected biclustering algorithm.

During execution, you will be prompted to select a biclustering algorithm and specify its parameters.

## Dependencies
- Python 3.8+
- Gensim
- NetworkX
- PyTorch Geometric
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm

Install these dependencies using `requirements.txt`.

## Contributing
Feel free to submit issues or contribute to this repository. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the Duke License.

