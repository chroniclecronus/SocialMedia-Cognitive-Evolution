# Evolutionary Dynamics of AI Discourse: A Multimodal Structural Entropy Approach

This repository contains the official source code for the paper "Evolutionary Dynamics of AI Discourse: A Multimodal Structural Entropy Approach" (Submitted to IEEE Transactions on Big Data).

## 📂 Project Structure

*   `src/`: Python source codes for graph construction and analysis.
*   `data/`: Sample dataset (Desensitized for privacy).
*   `output/`: Generated results.

## 🚀 How to Run

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the pipeline:
    ```bash
    # Step 1: Construct the graph
    python src/graph_construction.py --csv_path data/sample.csv

    # Step 2: Run evolutionary analysis
    python src/evolutionary_analysis.py --graph_path output/graph_v1/tv_hin_graph.pkl
    ```

## ⚙️ Theory Correspondence

| Code Parameter | Paper Symbol | Description |
| :--- | :---: | :--- |
| `threshold` | $\theta_{sem}$ | Semantic resolution threshold (0.80) |
| `stance_weight` | $1+\lambda$ | Stance modulation factor (1.2) |
| `compute_entropy` | $H^2(G)$ | 2D Structural Entropy |

## 📜 Citation

If you find this code and dataset useful, please cite our paper (wish me luck if my paper is accepted!).

Also, if you need the complete multimodal image and text dataset, you can contact me (of course, only after my paper is accepted).

