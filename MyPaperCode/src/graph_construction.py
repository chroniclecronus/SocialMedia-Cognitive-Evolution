"""
Social Media Multimodal Knowledge Evolution Analysis System
Module: Heterogeneous Graph Construction (TV-HIN)
Reference: "Evolutionary Dynamics of AI Discourse: A Multimodal Structural Entropy Approach"

This script handles:
1. Data Loading & Alignment (Text + LLaVA Embeddings)
2. Semantic Edge Construction (with Stance-Weighted Cosine Similarity)
3. Temporal Edge Construction (Time-Decay Dynamics)
4. User Interaction Edge Construction
5. Graph Export (Pickle, GraphML, GEXF)
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import logging
from tqdm import tqdm
import json
from scipy import sparse
import pickle

warnings.filterwarnings('ignore')

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration: Column Mapping ---
# NOTE: Update these values if your CSV headers are translated to English.
# Current setting assumes raw data still has Chinese headers.
COLUMN_MAPPING = {
    'id': '序号',
    'content': '微博内容',
    'timestamp': '发布时间',
    'stance': '立场',
    'uid': '用户UID',
    'verified_type': '认证类型',
    'image_id': '图片ID'
}


class WeiboDataLoader:
    """Module for loading raw data and aligning with multimodal embeddings."""

    def __init__(self, csv_path, feature_path):
        """
        Args:
            csv_path: Path to the metadata CSV file.
            feature_path: Path to the feature vector file (.pt or .npy).
        """
        self.csv_path = csv_path
        self.feature_path = feature_path
        self.data = None
        self.features = None

    def load_and_align(self):
        """Loads CSV and aligns it with feature vectors."""
        logger.info(f"Loading metadata from: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)

        # Check required columns
        required_cols = list(COLUMN_MAPPING.values())
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column in CSV: {col}")

        # Parse timestamps
        logger.info("Parsing timestamps...")
        time_col = COLUMN_MAPPING['timestamp']
        self.data[time_col] = pd.to_datetime(self.data[time_col], errors='coerce')

        # Sort by time
        self.data = self.data.sort_values(time_col).reset_index(drop=True)

        # Load Features
        logger.info(f"Loading feature vectors from: {self.feature_path}")
        if self.feature_path.endswith('.pt'):
            all_features = torch.load(self.feature_path)
            if isinstance(all_features, torch.Tensor):
                all_features = all_features.numpy()
        elif self.feature_path.endswith('.npy'):
            all_features = np.load(self.feature_path)
        else:
            raise ValueError("Unsupported feature format. Use .pt or .npy")

        logger.info(f"Raw feature shape: {all_features.shape}")

        # Alignment Logic
        # Assuming 'id' column in CSV corresponds to feature index (1-based index in CSV)
        feature_list = []
        valid_indices = []
        id_col = COLUMN_MAPPING['id']

        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Aligning features"):
            seq_num = int(row[id_col]) - 1  # Convert 1-based to 0-based index

            if seq_num < len(all_features):
                feature_list.append(all_features[seq_num])
                valid_indices.append(idx)
            else:
                logger.warning(f"Index {seq_num} out of bounds for feature matrix.")

        # Update data to keep only valid entries
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        self.features = np.array(feature_list)

        logger.info(f"Successfully loaded and aligned {len(self.data)} records.")
        return self.data, self.features

    def get_time_slices(self, interval='2h'):
        """Splits data into time windows."""
        logger.info(f"Slicing data by interval: {interval}...")

        time_col = COLUMN_MAPPING['timestamp']
        timestamps = self.data[time_col]

        if interval == '2h':
            delta = timedelta(hours=2)
        elif interval == '1d':
            delta = timedelta(days=1)
        elif interval == '7d':
            delta = timedelta(days=7)
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        time_slices = {}
        start_time = timestamps.min()
        current_slice_start = start_time
        current_slice_idx = 0

        for idx, ts in enumerate(timestamps):
            if ts >= current_slice_start + delta:
                current_slice_start = ts
                current_slice_idx += 1

            if current_slice_idx not in time_slices:
                time_slices[current_slice_idx] = []
            time_slices[current_slice_idx].append(idx)

        logger.info(f"Generated {len(time_slices)} time slices.")
        return time_slices


class SemanticEdgeBuilder:
    """
    Constructs semantic edges based on LLaVA embeddings.
    Implements Stance-Weighted Cosine Similarity (Eq. 3 in paper).
    """

    def __init__(self, threshold=0.80, top_k=30, stance_weight=1.2):
        """
        Args:
            threshold: Theta_sem, similarity threshold (Paper uses 0.8).
            top_k: Top-K neighbors for graph sparsity (Paper mentions 50, optimized to 30).
            stance_weight: Lambda factor for same-stance reinforcement (1 + 0.2 = 1.2).
        """
        self.threshold = threshold
        self.top_k = top_k
        self.stance_weight = stance_weight
        self.similarity_matrix = None

    def compute_similarity_matrix_batched(self, features, stances, batch_size=1000):
        """Computes cosine similarity in batches with stance weighting."""
        logger.info(f"Computing semantic similarity (Batch size: {batch_size})...")
        logger.info(f"Params: Threshold={self.threshold}, Top-K={self.top_k}, StanceWeight={self.stance_weight}")

        n_samples = len(features)
        rows, cols, data = [], [], []

        for i in tqdm(range(0, n_samples, batch_size), desc="Computing Batches"):
            i_end = min(i + batch_size, n_samples)
            batch_i = features[i:i_end]

            # Cosine similarity
            sim_batch = cosine_similarity(batch_i, features)

            for batch_idx, global_idx_i in enumerate(range(i, i_end)):
                for j in range(global_idx_i + 1, n_samples):
                    sim = sim_batch[batch_idx, j]

                    # Apply Stance Weighting (Eq. 3)
                    if stances[global_idx_i] == stances[j]:
                        sim_weighted = min(sim * self.stance_weight, 1.0)
                    else:
                        sim_weighted = sim

                    if sim_weighted > self.threshold:
                        rows.append(global_idx_i)
                        cols.append(j)
                        data.append(sim_weighted)

        self.similarity_matrix = sparse.coo_matrix(
            (data, (rows, cols)), shape=(n_samples, n_samples)
        )
        # Symmetrize
        self.similarity_matrix = self.similarity_matrix + self.similarity_matrix.T
        self.similarity_matrix = self.similarity_matrix.tocsr()

        logger.info(f"Similarity matrix constructed. NNZ elements: {self.similarity_matrix.nnz}")
        return self.similarity_matrix

    def build_edges_with_topk(self, features, stances):
        """Extracts edge list using Top-K sparsification."""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix_batched(features, stances)

        logger.info(f"Applying Top-K ({self.top_k}) sparsification...")
        n_nodes = self.similarity_matrix.shape[0]
        coo = self.similarity_matrix.tocoo()

        # Group neighbors
        neighbor_dict = {i: [] for i in range(n_nodes)}
        for i, j, w in zip(coo.row, coo.col, coo.data):
            if i != j:
                neighbor_dict[i].append((j, w))

        edges = []
        for i, neighbors in tqdm(neighbor_dict.items(), desc="Filtering Edges"):
            if neighbors:
                # Sort by weight desc, take top K
                neighbors.sort(key=lambda x: x[1], reverse=True)
                top_neighbors = neighbors[:self.top_k]

                for j, w in top_neighbors:
                    if i < j:  # Undirected graph, store once
                        edges.append((i, j, w, 'semantic'))

        logger.info(f"Generated {len(edges)} semantic edges.")
        return edges


class TemporalEdgeBuilder:
    """Constructs temporal edges with decay dynamics (Eq. 4 in paper)."""

    def __init__(self, window_hours=2):
        self.window_hours = window_hours
        self.window_seconds = window_hours * 3600

    def build_temporal_edges(self, timestamps, max_connections=100):
        logger.info(f"Constructing temporal edges (Window: {self.window_hours}h)...")

        n = len(timestamps)
        edges = []

        # Convert to unix timestamps for faster calc
        ts_seconds = np.array([t.timestamp() for t in timestamps])

        for i in tqdm(range(n), desc="Temporal Linking"):
            j = i + 1
            count = 0
            while j < n and count < max_connections:
                diff = ts_seconds[j] - ts_seconds[i]
                if diff <= self.window_seconds:
                    # Decay weight
                    weight = 1.0 - (diff / self.window_seconds)
                    edges.append((i, j, weight, 'temporal'))
                    count += 1
                else:
                    break
                j += 1

        logger.info(f"Generated {len(edges)} temporal edges.")
        return edges


class HeterogeneousGraph:
    """Manages the graph structure and export functions."""

    def __init__(self):
        self.graph = nx.Graph()

    def build(self, data_loader, semantic_edges, temporal_edges):
        logger.info("Assembling Heterogeneous Graph...")

        data = data_loader.data
        features = data_loader.features

        # 1. Add Nodes with Attributes
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Adding Nodes"):
            self.graph.add_node(
                idx,
                timestamp=row[COLUMN_MAPPING['timestamp']],
                stance=row.get(COLUMN_MAPPING['stance'], 'unknown'),
                uid=row[COLUMN_MAPPING['uid']],
                # Truncate content for GEXF compatibility
                content=str(row[COLUMN_MAPPING['content']])[:100]
            )

        # 2. Add Semantic Edges
        logger.info(f"Adding {len(semantic_edges)} semantic edges...")
        self.graph.add_edges_from([
            (u, v, {'weight': w, 'edge_type': t}) for u, v, w, t in semantic_edges
        ])

        # 3. Add Temporal Edges
        logger.info(f"Adding {len(temporal_edges)} temporal edges...")
        self.graph.add_edges_from([
            (u, v, {'weight': w, 'edge_type': t}) for u, v, w, t in temporal_edges
        ])

        # 4. Add User Continuity Edges (Simple heuristic)
        self._add_user_edges(data)

        self._print_stats()

    def _add_user_edges(self, data, max_edges=5):
        """Links posts from the same user (User Continuity Layer)."""
        logger.info("Adding user continuity edges...")
        user_groups = data.groupby(COLUMN_MAPPING['uid']).indices
        edges = []

        for uid, indices in user_groups.items():
            if len(indices) > 1:
                # Sort indices by time (already sorted in data)
                # Link sequential posts
                for k in range(len(indices) - 1):
                    # Limit to immediate next few posts to avoid dense cliques
                    for m in range(1, min(max_edges, len(indices) - k)):
                        u, v = indices[k], indices[k + m]
                        edges.append((u, v, 1.0, 'user_interaction'))

        self.graph.add_edges_from([
            (u, v, {'weight': w, 'edge_type': t}) for u, v, w, t in edges
        ])
        logger.info(f"Added {len(edges)} user edges.")

    def _print_stats(self):
        n = self.graph.number_of_nodes()
        e = self.graph.number_of_edges()
        logger.info("Graph Statistics:")
        logger.info(f"  Nodes: {n}")
        logger.info(f"  Edges: {e}")
        logger.info(f"  Avg Degree: {2 * e / n:.2f}")

    def save(self, output_dir, filename="tv_hin_graph"):
        base_path = os.path.join(output_dir, filename)

        # 1. Pickle (Complete Data)
        nx.write_gpickle(self.graph, f"{base_path}.pkl")

        # 2. GraphML (Standard)
        # Note: Removing complex types if necessary for GraphML
        nx.write_graphml(self.graph, f"{base_path}.graphml")

        logger.info(f"Graph saved to {base_path}.pkl / .graphml")


def parse_args():
    parser = argparse.ArgumentParser(description="TV-HIN Graph Construction Pipeline")

    # Paths (Default to relative paths for reproducibility)
    parser.add_argument('--csv_path', type=str, default='./data/sample_data.csv',
                        help='Path to the metadata CSV file')
    parser.add_argument('--feature_path', type=str, default='./data/embeddings.pt',
                        help='Path to the LLaVA embedding file')
    parser.add_argument('--output_dir', type=str, default='./output/graph',
                        help='Directory to save the constructed graph')

    # Parameters
    parser.add_argument('--threshold', type=float, default=0.80, help='Semantic similarity threshold')
    parser.add_argument('--top_k', type=int, default=30, help='Top-K neighbors for semantic edges')
    parser.add_argument('--stance_weight', type=float, default=1.2, help='Weight multiplier for same-stance edges')
    parser.add_argument('--time_window', type=int, default=2, help='Time decay window in hours')

    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting Graph Construction Pipeline")
    logger.info(f"Config: Threshold={args.threshold}, TopK={args.top_k}, TimeWindow={args.time_window}h")
    logger.info("=" * 60)

    try:
        # 1. Data Loading
        loader = WeiboDataLoader(args.csv_path, args.feature_path)
        data, features = loader.load_and_align()

        # 2. Semantic Edges
        stances = data[COLUMN_MAPPING['stance']].tolist()
        sem_builder = SemanticEdgeBuilder(args.threshold, args.top_k, args.stance_weight)
        sem_edges = sem_builder.build_edges_with_topk(features, stances)

        # 3. Temporal Edges
        timestamps = data[COLUMN_MAPPING['timestamp']].tolist()
        temp_builder = TemporalEdgeBuilder(args.time_window)
        temp_edges = temp_builder.build_temporal_edges(timestamps)

        # 4. Graph Assembly & Export
        hin_graph = HeterogeneousGraph()
        hin_graph.build(loader, sem_edges, temp_edges)
        hin_graph.save(args.output_dir)

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()