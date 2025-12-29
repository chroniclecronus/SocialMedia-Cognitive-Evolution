"""
Social Media Multimodal Knowledge Evolution Analysis System
Module: Network Topology Visualization
Reference: "Evolutionary Dynamics of AI Discourse", Figure 7

Description:
    Visualizes the TV-HIN snapshots using Force-Directed (Spring) Layout.
    Colors nodes by stance (Support/Oppose/Neutral).
"""

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pickle
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

# Visualization Config
STANCE_COLORS = {
    '支持': '#2ca02c', 'support': '#2ca02c',  # Green
    '反对': '#d62728', 'oppose': '#d62728',  # Red
    '中立': '#1f77b4', 'neutral': '#1f77b4',  # Blue
    'unknown': '#7f7f7f'  # Gray
}


class TopologyVisualizer:
    def __init__(self, graph_path, community_dir, output_dir):
        self.graph_path = graph_path
        self.community_dir = community_dir
        self.output_dir = output_dir
        self.full_graph = None
        os.makedirs(output_dir, exist_ok=True)

    def load_graph(self):
        print(f"Loading full graph from {self.graph_path}...")
        try:
            with open(self.graph_path, 'rb') as f:
                self.full_graph = pickle.load(f)
            print(f"Graph loaded. Nodes: {self.full_graph.number_of_nodes()}")
        except Exception as e:
            print(f"Error loading graph: {e}")
            exit(1)

    def load_slice_meta(self, slice_id):
        """Loads node stance/community info for a specific slice."""
        # Try different naming conventions from previous steps
        candidates = [
            f"community_structure_slice_{slice_id}.csv",
            f"figure7_community_structure_slice_{slice_id}.csv"
        ]

        df = None
        for fname in candidates:
            path = os.path.join(self.community_dir, fname)
            if os.path.exists(path):
                # Try reading with/without header
                try:
                    df = pd.read_csv(path)
                    # Standardize columns
                    if len(df.columns) == 3:
                        df.columns = ['node_id', 'community_id', 'stance']
                    elif len(df.columns) >= 4:
                        df.columns = ['slice_id', 'node_id', 'community_id', 'stance']
                    break
                except:
                    continue

        if df is None:
            print(f"Warning: Metadata for slice {slice_id} not found.")
            return None

        return dict(zip(df['node_id'], df['stance']))

    def visualize_slice(self, slice_id, title_suffix=""):
        """Draws the network for a specific time slice."""
        print(f"Visualizing Slice {slice_id}...")

        node_stance_map = self.load_slice_meta(slice_id)
        if not node_stance_map: return

        # Extract Subgraph
        nodes = list(node_stance_map.keys())
        subgraph = self.full_graph.subgraph(nodes)

        # Color Map
        node_colors = []
        for n in subgraph.nodes():
            raw_stance = node_stance_map.get(n, 'neutral')
            # Handle potential Chinese/English mix
            stance_key = str(raw_stance).strip()
            node_colors.append(STANCE_COLORS.get(stance_key, STANCE_COLORS['unknown']))

        # Layout (Computationally Expensive)
        print("  Computing Spring Layout...")
        pos = nx.spring_layout(subgraph, k=0.15, iterations=50, seed=42)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 12))

        # Draw Edges (Very transparent)
        nx.draw_networkx_edges(subgraph, pos, ax=ax, alpha=0.05, width=0.5, edge_color='#cccccc')

        # Draw Nodes
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=node_colors, node_size=20, alpha=0.8)

        # Decoration
        ax.set_title(f"Network Topology - Slice {slice_id}\n{title_suffix}", fontsize=16)
        ax.axis('off')

        # Legend
        legend_elements = [
            Patch(facecolor=STANCE_COLORS['support'], label='Support'),
            Patch(facecolor=STANCE_COLORS['oppose'], label='Oppose'),
            Patch(facecolor=STANCE_COLORS['neutral'], label='Neutral')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Save
        out_path = os.path.join(self.output_dir, f"Topology_Slice_{slice_id}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {out_path}")
        plt.close()

    def run_pipeline(self):
        self.load_graph()
        # Visualize Key Moments (Inflation -> Cooling -> Equilibrium)
        target_slices = [20, 60, 90, 120]

        for sid in target_slices:
            self.visualize_slice(sid)


def main():
    parser = argparse.ArgumentParser(description="Network Topology Visualizer")
    parser.add_argument('--graph_path', type=str, required=True,
                        help='Path to .pkl graph file')
    parser.add_argument('--community_dir', type=str, required=True,
                        help='Directory containing community structure CSVs')
    parser.add_argument('--output_dir', type=str, default='./output/topology_viz')
    args = parser.parse_args()

    viz = TopologyVisualizer(args.graph_path, args.community_dir, args.output_dir)
    viz.run_pipeline()


if __name__ == "__main__":
    main()