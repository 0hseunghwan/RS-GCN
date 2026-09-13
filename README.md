```markdown
# RS-GCN

A paper recommendation project that uses embeddings learned by a Graph Convolutional Network (GCN) to recommend similar papers. It trains a GCN on the [Cora](https://paperswithcode.com/dataset/cora) citation network dataset, then serves a recommendation API (FastAPI) based on cosine similarity between the learned embeddings.

## Components

- `models/gcn.py` — GCN model definitions
  - `GCN`: a 2-layer GCN for node classification (`log_softmax` output)
  - `GCNEncoder`: an embedding encoder for the recommender (no classification head, outputs embeddings only)
- `train.py` — Trains the `GCN` node-classification model on the Cora dataset and evaluates accuracy
- `train_RS.py` — Trains `GCNEncoder` to produce paper embeddings and generates the data the recommender needs (`data/embeddings.npy`, `data/papers.json`, `data/edges.json`)
- `RS.py` — `PaperRecommender` class that loads the trained embeddings and performs keyword/category-based similar-paper search
- `app.py` — FastAPI application that serves the recommender
- `requirements.txt` — Dependency list

## Installation

```bash
pip install -r requirements.txt
```

> `torch`, `torch-geometric`, and their extensions (`torch_scatter`, `torch_sparse`) may need builds matching your PyTorch/CUDA version. Depending on your environment, install them separately following the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Usage

### 1. Train the GCN node classifier (optional)

Check baseline GCN classification performance on the Cora dataset.

```bash
python train.py
```

### 2. Generate embeddings for the recommender (required)

Trains `GCNEncoder` and generates paper embeddings and metadata under `data/`. This must be run before starting the recommendation API.

```bash
python train_RS.py
```

Once finished, the following files are created:

- `data/embeddings.npy` — GCN embeddings for each paper (node)
- `data/papers.json` — Paper metadata (id, title, label, category)
- `data/edges.json` — List of citation edges

### 3. Run the recommendation API server

```bash
python app.py
```

The server runs on `http://localhost:8000` by default.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Main page (`static/index.html`) |
| GET | `/api/search?q={query}&top_k={n}` | Search for similar papers by keyword/category |
| GET | `/api/paper/{paper_id}` | Details for a specific paper plus similar papers |
| GET | `/api/categories` | List of all categories with paper counts |
| GET | `/api/graph?limit={n}` | Node/edge graph data for visualization |

Example:

```bash
curl "http://localhost:8000/api/search?q=neural+networks&top_k=5"
```

## Dataset

Uses the [Planetoid Cora](https://paperswithcode.com/dataset/cora) dataset, classified into 7 categories:

- Case Based
- Genetic Algorithms
- Neural Networks
- Probabilistic Methods
- Reinforce Learning
- Rule Learning
- Theory

Since Cora does not include real paper titles, `train_RS.py` generates placeholder titles based on each paper's category.
```
