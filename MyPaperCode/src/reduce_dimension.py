"""
Social Media Multimodal Knowledge Evolution Analysis System
Module: Dimensionality Reduction (PCA)
Reference: "Evolutionary Dynamics of AI Discourse", Section III.A.2

Description:
    Projects ultra-high-dimensional raw features (R^D_high, D~3M)
    onto a lower-dimensional Riemannian manifold (R^d, d=512)
    using Principal Component Analysis.
"""

import os
import argparse
import torch
import numpy as np
import glob
import gc
import logging
from sklearn.decomposition import PCA
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_and_flatten_tensor(file_path):
    """
    Loads a .pt file and flattens it to [1, 2949120].
    Shape transformation: [1, 576, 5120] -> [1, 2949120]
    """
    try:
        data = torch.load(file_path, map_location='cpu')

        # Handle dict wrapper if present
        if isinstance(data, dict):
            tensor = list(data.values())[0]
        elif isinstance(data, torch.Tensor):
            tensor = data
        else:
            return None

        # Flatten logic
        if len(tensor.shape) == 3:
            tensor_flat = tensor.view(1, -1)
        elif len(tensor.shape) == 2:
            tensor_flat = tensor
        else:
            return None

        data_np = tensor_flat.numpy()

        # Sanity check
        if np.isnan(data_np).any() or np.isinf(data_np).any():
            return None

        return data_np
    except Exception:
        return None


def collect_training_data(input_dir, limit=5000):
    """
    Collects a random subset of data to train the PCA model.
    Loading ALL data might cause OOM (Out of Memory).
    """
    logger.info(f"Phase 1: Collecting data for PCA training (Limit: {limit} samples)...")

    all_files = glob.glob(os.path.join(input_dir, "**/*.pt"), recursive=True)

    if not all_files:
        raise ValueError(f"No .pt files found in {input_dir}")

    # Shuffle files to get a representative sample
    np.random.shuffle(all_files)

    training_data = []
    count = 0

    for pt_file in tqdm(all_files, desc="Loading Training Samples"):
        data = load_and_flatten_tensor(pt_file)
        if data is not None:
            training_data.append(data)
            count += 1
            if limit and count >= limit:
                break

    if not training_data:
        raise ValueError("Failed to load any valid data.")

    combined = np.vstack(training_data)
    logger.info(f"Training set shape: {combined.shape}")
    return combined


def train_pca(data, target_dim=512):
    """Trains the PCA model."""
    logger.info(f"Phase 2: Training PCA model (Target Dim: {target_dim})...")

    pca = PCA(n_components=target_dim, random_state=42)
    pca.fit(data)

    explained_variance = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA Training Complete.")
    logger.info(f"Total Explained Variance Ratio: {explained_variance:.4f}")

    return pca


def transform_and_save(input_dir, output_dir, pca_model, target_dim=512):
    """Applies PCA transform to all files and saves them."""
    logger.info("Phase 3: Transforming and saving all files...")

    all_files = glob.glob(os.path.join(input_dir, "**/*.pt"), recursive=True)
    success = 0

    for pt_file in tqdm(all_files, desc="Transforming"):
        try:
            # 1. Load
            data = load_and_flatten_tensor(pt_file)
            if data is None:
                continue

            # 2. Transform
            reduced = pca_model.transform(data)  # [1, 512]

            # 3. Post-process (Ensure exact dimension)
            if reduced.shape[1] != target_dim:
                # Pad with zeros if necessary (rare edge case)
                padding = np.zeros((1, target_dim - reduced.shape[1]))
                reduced = np.concatenate([reduced, padding], axis=1)

            # 4. Save
            # Maintain directory structure relative to input_dir
            rel_path = os.path.relpath(pt_file, input_dir)
            save_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(torch.from_numpy(reduced), save_path)
            success += 1

        except Exception as e:
            logger.warning(f"Failed to process {pt_file}: {e}")

    logger.info(f"Processing complete. Successfully transformed {success}/{len(all_files)} files.")


def main(args):
    # 1. Train PCA
    try:
        training_data = collect_training_data(args.input_dir, limit=args.train_limit)
        pca_model = train_pca(training_data, target_dim=args.target_dim)

        # Free memory
        del training_data
        gc.collect()

        # 2. Transform All Data
        transform_and_save(args.input_dir, args.output_dir, pca_model, target_dim=args.target_dim)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Dimensional Feature Reduction (PCA)")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing raw .pt files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save reduced .pt files")
    parser.add_argument("--target-dim", type=int, default=512, help="Target dimension (d)")
    parser.add_argument("--train-limit", type=int, default=5000,
                        help="Number of samples to use for PCA training (to save RAM)")

    args = parser.parse_args()
    main(args)