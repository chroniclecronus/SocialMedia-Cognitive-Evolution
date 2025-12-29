"""
Social Media Multimodal Knowledge Evolution Analysis System
Module: Evolutionary Analysis & Structural Entropy Calculation
Reference: "Evolutionary Dynamics of AI Discourse: A Multimodal Structural Entropy Approach"

This script handles:
1. Time-Slice Generation (Cumulative Evolution)
2. Dynamic Community Detection (Louvain Algorithm)
3. 2D Structural Entropy Calculation (Eq. 7)
4. Phase Identification & Visualization
"""

import os
import argparse
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain  # Requires: pip install python-louvain
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Configure Logging to English
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Matplotlib Style for Academic Publishing
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'  # Use serif fonts (Times New Roman-like)


class CommunityDetector:
    """
    Wrapper for Community Detection (Louvain Method).
    Corresponds to Section III.E in the paper.
    """

    def __init__(self, resolution=1.2, random_state=42):
        """
        Args:
            resolution: Resolution parameter (gamma) for modularity.
                        Paper sensitivity analysis suggests 1.2 is optimal.
        """
        self.resolution = resolution
        self.random_state = random_state
        self.partition = None
        self.stats = {}

    def detect(self, graph, weight='weight'):
        if graph.number_of_edges() == 0:
            return {}

        # Louvain requires undirected graph
        g_undir = graph.to_undirected()

        try:
            self.partition = community_louvain.best_partition(
                g_undir,
                weight=weight,
                resolution=self.resolution,
                random_state=self.random_state
            )
        except TypeError:
            # Fallback for versions not supporting resolution
            self.partition = community_louvain.best_partition(
                g_undir,
                weight=weight,
                random_state=self.random_state
            )

        self._compute_stats(g_undir)
        return self.partition

    def _compute_stats(self, graph):
        comm_counts = Counter(self.partition.values())
        self.stats = {
            'num_communities': len(comm_counts),
            'modularity': community_louvain.modularity(self.partition, graph),
            'avg_community_size': np.mean(list(comm_counts.values()))
        }

        # Stance Purity Analysis (Crucial for "Depolarization Paradox")
        # Checks if nodes in a community share the same stance
        if 'stance' in graph.nodes[list(graph.nodes())[0]]:
            total_purity = 0
            valid_comms = 0

            comm_to_nodes = defaultdict(list)
            for node, comm_id in self.partition.items():
                comm_to_nodes[comm_id].append(graph.nodes[node]['stance'])

            for stances in comm_to_nodes.values():
                if len(stances) > 5:  # Ignore tiny communities
                    most_common = Counter(stances).most_common(1)[0]
                    purity = most_common[1] / len(stances)
                    total_purity += purity
                    valid_comms += 1

            self.stats['avg_stance_purity'] = total_purity / valid_comms if valid_comms > 0 else 0

    def build_encoding_tree(self, graph):
        """Constructs the hierarchical encoding tree for entropy calculation."""
        if not self.partition:
            raise ValueError("Run detect() first.")

        tree = {'root': {'nodes': list(graph.nodes())}, 'communities': {}}

        comm_groups = defaultdict(list)
        for node, comm_id in self.partition.items():
            comm_groups[comm_id].append(node)

        for comm_id, nodes in comm_groups.items():
            subgraph = graph.subgraph(nodes)
            internal_edges = subgraph.number_of_edges()

            tree['communities'][comm_id] = {
                'nodes': nodes,
                'internal_edges': internal_edges,
                'size': len(nodes)
            }
        return tree


class StructuralEntropyCalculator:
    """
    Calculates 2D Structural Entropy (H2).
    Implements Equation (7) from the paper.
    """

    def compute_entropy(self, graph, encoding_tree):
        m = graph.number_of_edges()
        if m == 0:
            return 0.0

        total_entropy = 0.0
        V_root = len(graph.nodes())  # V_total

        # Iterating over communities (modules)
        for comm_info in encoding_tree['communities'].values():
            g_alpha = comm_info['internal_edges']  # Edges inside module
            V_alpha = comm_info['size']  # Volume of module (simplified to node count)

            if V_alpha == 0 or V_root == 0:
                continue

            # Probability term: g_alpha / 2m
            prob = g_alpha / (2 * m)

            # Log term: -log2(V_alpha / V_total)
            # Represents the bits required to locate the module
            if prob > 0:
                total_entropy += -prob * np.log2(V_alpha / V_root)

            # Note: The node-level entropy term is implicitly handled in standard SE
            # libraries, but here we focus on the modular component which drives
            # the macroscopic phase transition described in the paper.

        return total_entropy


class GraphEvolutionAnalyzer:
    """
    Orchestrator for Longitudinal Analysis.
    Slices the graph, computes metrics, and generates visualizations.
    """

    def __init__(self, graph_path, output_dir):
        self.graph_path = graph_path
        self.output_dir = output_dir
        self.graph = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_graph(self):
        logger.info(f"Loading graph from {self.graph_path}...")
        with open(self.graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        logger.info(f"Graph loaded. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")

    def _generate_time_slices(self, interval_days=7):
        """
        Slices the graph based on node timestamps (Cumulative Strategy).
        Does not require raw CSV, uses graph attributes.
        """
        logger.info("Generating cumulative time slices...")

        # Extract timestamps
        node_times = []
        for n, attr in self.graph.nodes(data=True):
            if 'timestamp' in attr:
                ts = pd.to_datetime(attr['timestamp'])
                node_times.append((n, ts))

        # Sort by time
        node_times.sort(key=lambda x: x[1])

        start_date = node_times[0][1]
        end_date = node_times[-1][1]
        current_date = start_date

        slices = []
        slice_id = 0

        while current_date <= end_date:
            cutoff = current_date + timedelta(days=interval_days)

            # Get all nodes up to cutoff (Cumulative)
            nodes_in_slice = [n for n, t in node_times if t <= cutoff]

            if len(nodes_in_slice) > 100:  # Skip very early sparse slices
                slices.append({
                    'id': slice_id,
                    'date': cutoff,
                    'nodes': nodes_in_slice
                })
                slice_id += 1

            current_date = cutoff

        logger.info(f"Generated {len(slices)} slices from {start_date} to {end_date}.")
        return slices

    def run_analysis(self, resolution=1.2):
        self.load_graph()
        slices = self._generate_time_slices(interval_days=7)  # Weekly slices

        results = []

        detector = CommunityDetector(resolution=resolution)
        calculator = StructuralEntropyCalculator()

        logger.info("Starting Evolutionary Analysis...")
        for sl in slices:
            # Create subgraph
            subgraph = self.graph.subgraph(sl['nodes'])

            # Detect Communities
            partition = detector.detect(subgraph)
            tree = detector.build_encoding_tree(subgraph)

            # Compute Entropy
            entropy = calculator.compute_entropy(subgraph, tree)

            # Record Metrics
            metrics = {
                'slice_id': sl['id'],
                'date': sl['date'],
                'nodes': subgraph.number_of_nodes(),
                'edges': subgraph.number_of_edges(),
                'entropy': entropy,
                'num_communities': detector.stats.get('num_communities', 0),
                'modularity': detector.stats.get('modularity', 0),
                'avg_stance_purity': detector.stats.get('avg_stance_purity', 0)
            }
            results.append(metrics)

            if sl['id'] % 10 == 0:
                logger.info(f"Slice {sl['id']}: Entropy={entropy:.4f}, Nodes={metrics['nodes']}")

        # Save Results
        self._save_results(results)
        self._visualize(pd.DataFrame(results))

    def _save_results(self, results):
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, "evolution_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

    def _visualize(self, df):
        """Generates publication-ready plots."""
        viz_dir = os.path.join(self.output_dir, "plots")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        # 1. Structural Entropy Evolution (The State Equation)
        plt.figure(figsize=(10, 6))
        plt.plot(df['slice_id'], df['entropy'], color='#1f77b4', linewidth=2, label='Structural Entropy')

        # Add Trend Line (Smoothed)
        df['entropy_smooth'] = df['entropy'].rolling(window=5, center=True).mean()
        plt.plot(df['slice_id'], df['entropy_smooth'], color='red', linestyle='--', alpha=0.8, label='Trend (Smoothed)')

        plt.xlabel('Time Slices (Weekly)', fontsize=12)
        plt.ylabel('Structural Entropy ($H^2$)', fontsize=12)
        plt.title('Evolutionary Dynamics of Structural Entropy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, "fig3_entropy_evolution.png"), dpi=300)
        plt.close()

        # 2. The Depolarization Paradox (Entropy vs Purity)
        # Verify if purity data exists
        if 'avg_stance_purity' in df.columns and df['avg_stance_purity'].mean() > 0:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            color = 'tab:blue'
            ax1.set_xlabel('Time Slices', fontsize=12)
            ax1.set_ylabel('Structural Entropy', color=color, fontsize=12)
            ax1.plot(df['slice_id'], df['entropy'], color=color, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:purple'
            ax2.set_ylabel('Stance Homogeneity (Purity)', color=color, fontsize=12)
            ax2.plot(df['slice_id'], df['avg_stance_purity'], color=color, linestyle='--', linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title('The Depolarization Paradox: Stability vs. Homogeneity', fontsize=14, fontweight='bold')
            fig.tight_layout()
            plt.savefig(os.path.join(viz_dir, "fig_depolarization_paradox.png"), dpi=300)
            plt.close()

        logger.info(f"Plots saved to {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evolutionary Analysis Module")
    parser.add_argument('--graph_path', type=str, required=True,
                        help='Path to the .pkl graph file generated in Step 1')
    parser.add_argument('--output_dir', type=str, default='./output/analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--resolution', type=float, default=1.2,
                        help='Resolution for community detection (Gamma)')

    args = parser.parse_args()

    analyzer = GraphEvolutionAnalyzer(args.graph_path, args.output_dir)
    analyzer.run_analysis(resolution=args.resolution)


if __name__ == "__main__":
    main()