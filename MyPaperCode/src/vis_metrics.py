"""
Social Media Multimodal Knowledge Evolution Analysis System
Module: Statistical Visualization & Table Generation
Reference: "Evolutionary Dynamics of AI Discourse"

Generates:
- Figure : Structural Entropy Evolution (The State Equation)
- Figure : Phase Transition (Entropy vs Communities)
- Figure : Depolarization Paradox (Homogeneity Decline)
- Figure : Computational Performance
- Tables : Statistical Summaries
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# IEEE Standard Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class MetricVisualizer:
    def __init__(self, json_path, output_dir):
        self.json_path = json_path
        self.output_dir = output_dir
        self.df = self._load_data()
        os.makedirs(output_dir, exist_ok=True)

    def _load_data(self):
        print(f"Loading metrics from {self.json_path}...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Ensure slice_id is sorted
        df = df.sort_values('slice_id')
        return df

    def plot_fig1_entropy_evolution(self):
        """Figure 1: Structural Entropy Evolution with Phases."""
        print("Generating Figure 1 (Entropy Evolution)...")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot Line
        ax.plot(self.df['slice_id'], self.df['entropy'],
                linewidth=2.5, color='#1f77b4', label='Structural Entropy ($H^2$)')

        # Add Phases Background
        # Assuming phases at 30 and 70 based on your previous code
        phases = [(0, 30, '#ff7f0e', 'Phase I: Inflation'),
                  (30, 70, '#d62728', 'Phase II: Cooling'),
                  (70, 120, '#2ca02c', 'Phase III: Equilibrium')]

        for start, end, color, label in phases:
            ax.axvspan(start, end, alpha=0.1, color=color, label=label)

        # Annotate Max Entropy
        max_idx = self.df['entropy'].idxmax()
        max_val = self.df.loc[max_idx, 'entropy']
        max_slice = self.df.loc[max_idx, 'slice_id']

        ax.annotate(f'Max Entropy\n({max_val:.3f})',
                    xy=(max_slice, max_val), xytext=(max_slice + 10, max_val + 0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05))

        ax.set_xlabel('Time Slice (Weekly)')
        ax.set_ylabel('Structural Entropy')
        ax.set_title('Evolutionary Dynamics of Cognitive Structure')
        ax.legend(loc='upper right')
        ax.set_xlim(0, self.df['slice_id'].max())

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Fig1_Entropy_Evolution.png'), dpi=300)
        plt.close()

    def plot_fig2_entropy_vs_communities(self):
        """Figure 2: Dual Axis Plot (Entropy vs Community Count)."""
        print("Generating Figure 2 (Entropy vs Communities)...")
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1 = '#1f77b4'
        ax1.set_xlabel('Time Slice')
        ax1.set_ylabel('Structural Entropy', color=color1)
        ax1.plot(self.df['slice_id'], self.df['entropy'], color=color1, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = '#d62728'
        ax2.set_ylabel('Number of Communities', color=color2)
        ax2.plot(self.df['slice_id'], self.df['communities'], color=color2, linestyle='--', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title('Differentiation-Integration Process')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Fig2_Entropy_vs_Communities.png'), dpi=300)
        plt.close()

    def plot_fig3_depolarization(self):
        """Figure 3: The Depolarization Paradox (Stance Purity Decline)."""
        if 'avg_purity' not in self.df.columns:
            print("Skipping Fig 3: 'avg_purity' column missing.")
            return

        print("Generating Figure 3 (Depolarization Paradox)...")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Raw Data
        ax.scatter(self.df['slice_id'], self.df['avg_purity'], alpha=0.3, color='gray', label='Raw Data')

        # Polynomial Fit
        X = self.df['slice_id'].values.reshape(-1, 1)
        y = self.df['avg_purity'].values
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)

        ax.plot(self.df['slice_id'], y_pred, color='#9467bd', linewidth=3, label='Quadratic Trend')

        ax.set_xlabel('Time Slice')
        ax.set_ylabel('Stance Homogeneity (Purity)')
        ax.set_title('The Depolarization Paradox: Decline of Echo Chambers')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Fig3_Depolarization_Paradox.png'), dpi=300)
        plt.close()

    def plot_fig8_performance(self):
        """Figure 8: Computational Performance Analysis."""
        print("Generating Figure 8 (Performance)...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Time vs Nodes
        ax1.scatter(self.df['nodes'], self.df['computation_time'], alpha=0.6)

        # Linear fit for complexity check
        X = self.df['nodes'].values.reshape(-1, 1)
        y = self.df['computation_time'].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        ax1.plot(self.df['nodes'], y_pred, color='red', linestyle='--', label=f'Fit ($R^2={model.score(X, y):.2f}$)')
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Computation Time (s)')
        ax1.set_title('Scalability: Time vs Nodes')
        ax1.legend()

        # Time vs Slice
        ax2.plot(self.df['slice_id'], self.df['computation_time'], color='green')
        ax2.set_xlabel('Time Slice')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Processing Time per Slice')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Fig8_Performance.png'), dpi=300)
        plt.close()

    def generate_tables(self):
        """Generates summary tables (CSV/Excel)."""
        print("Generating Tables 1-5...")

        # Table 1: Basic Stats Summary
        t1 = self.df[['slice_id', 'nodes', 'edges', 'communities', 'entropy', 'modularity']].describe().T
        t1.to_csv(os.path.join(self.output_dir, 'Table1_Descriptive_Stats.csv'))

        # Table 4: Phase Stats
        # Simple heuristic mapping for phases
        def get_phase(sid):
            if sid <= 30: return 'Inflation'
            if sid <= 70: return 'Cooling'
            return 'Equilibrium'

        self.df['Phase'] = self.df['slice_id'].apply(get_phase)
        t4 = self.df.groupby('Phase')[['entropy', 'modularity', 'avg_purity', 'communities']].mean()
        t4.to_csv(os.path.join(self.output_dir, 'Table4_Phase_Analysis.csv'))

        print("Tables saved to output directory.")


def main():
    parser = argparse.ArgumentParser(description="Statistical Visualization Module")
    parser.add_argument('--json_path', type=str,
                        default='./day3_4_output/full_computation/full_computation_results.json',
                        help='Path to the results JSON file')
    parser.add_argument('--output_dir', type=str, default='./output/figures_and_tables',
                        help='Directory to save figures')
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: File not found: {args.json_path}")
        return

    viz = MetricVisualizer(args.json_path, args.output_dir)

    # Generate Plots
    viz.plot_fig1_entropy_evolution()
    viz.plot_fig2_entropy_vs_communities()
    viz.plot_fig3_depolarization()
    viz.plot_fig8_performance()

    # Generate Tables
    viz.generate_tables()

    print("All statistical visualizations completed.")


if __name__ == "__main__":
    main()