\# Evolutionary Dynamics of AI Discourse: A Multimodal Structural Entropy Approach



This repository contains the official source code for the paper \*\*"Evolutionary Dynamics of AI Discourse: A Multimodal Structural Entropy Approach"\*\* (Submitted to IEEE Transactions on Big Data).



\## 📂 Project Structure



\*   `src/`: Python source codes for graph construction and analysis.

\*   `data/`: Sample dataset (Desensitized for privacy).

\*   `output/`: Generated results.



\## 🚀 How to Run



1\.  \*\*Install dependencies:\*\*

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



2\.  \*\*Run the pipeline:\*\*

&nbsp;   ```bash

&nbsp;   # Step 1: Construct the graph

&nbsp;   python src/graph\_construction.py --csv\_path data/sample.csv



&nbsp;   # Step 2: Run evolutionary analysis

&nbsp;   python src/evolutionary\_analysis.py --graph\_path output/graph\_v1/tv\_hin\_graph.pkl

&nbsp;   ```



\## ⚙️ Theory Correspondence



| Code Parameter | Paper Symbol | Description |

| :--- | :---: | :--- |

| `--threshold` | $\\theta\_{sem}$ | Semantic resolution threshold (0.80) |

| `--stance\_weight` | $1+\\lambda$ | Stance modulation factor (1.2) |

| `compute\_entropy` | $H^2(G)$ | 2D Structural Entropy |



\## 📜 Citation



If you find this code useful, please cite our paper:



```bibtex

@article{Zhang2025Evolutionary,

&nbsp; title={Evolutionary Dynamics of AI Discourse},

&nbsp; journal={IEEE Transactions on Big Data (Submitted)},

&nbsp; year={2025}

}

