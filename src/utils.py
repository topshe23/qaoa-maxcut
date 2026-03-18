# src/utils.py
# Plotting and saving helpers

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import os


def plot_convergence(cut_history, optimal_cut, max_cut, save_path):
    """Plots QAOA convergence — expected cut vs iteration."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(cut_history, color='steelblue', linewidth=2, label='Expected Cut Value')
    ax.axhline(y=optimal_cut, color='red', linestyle='--',
               linewidth=1.8, label=f'QAOA converged: {optimal_cut:.4f}')
    ax.axhline(y=max_cut, color='green', linestyle=':',
               linewidth=1.8, label=f'Optimal MaxCut: {max_cut}')

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Expected Cut Value', fontsize=13)
    ax.set_title('QAOA Convergence — MaxCut Optimization', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_measurement_distribution(counts, G, best_bitstring, save_path, top_n=10):
    """Plots top N most measured bitstrings and their cut values."""
    # Get top N bitstrings by count
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    bitstrings = [b[::-1] for b, _ in sorted_counts]
    frequencies = [c / sum(counts.values()) for _, c in sorted_counts]
    cut_values = [compute_cut(b, G) for b in bitstrings]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['green' if b == best_bitstring else
              'steelblue' if c == max(cut_values) else
              'lightsteelblue' for b, c in zip(bitstrings, cut_values)]

    bars = ax.bar(range(len(bitstrings)), frequencies, color=colors, edgecolor='white')

    ax.set_xticks(range(len(bitstrings)))
    ax.set_xticklabels([f'{b}\n(cut={c})' for b, c in zip(bitstrings, cut_values)],
                       fontsize=8, rotation=45)
    ax.set_xlabel('Bitstring (Partition)', fontsize=13)
    ax.set_ylabel('Measurement Probability', fontsize=13)
    ax.set_title('QAOA Measurement Distribution — Top Solutions', fontsize=15, fontweight='bold')

    best_patch = mpatches.Patch(color='green', label='Best solution found')
    good_patch = mpatches.Patch(color='steelblue', label='High cut value')
    ax.legend(handles=[best_patch, good_patch], fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_graph_partition(G, best_bitstring, save_path):
    """Visualizes the graph with nodes colored by partition."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pos = nx.spring_layout(G, seed=42)
    colors_partition = ['steelblue' if bit == '0' else 'darkorange'
                        for bit in best_bitstring]

    # Left: partitioned graph
    nx.draw_networkx(G, pos, ax=axes[0], node_color=colors_partition,
                     node_size=800, font_color='white', font_weight='bold',
                     edge_color='gray', width=2)

    cut_edges = [(u, v) for u, v in G.edges()
                 if best_bitstring[u] != best_bitstring[v]]
    non_cut_edges = [(u, v) for u, v in G.edges()
                     if best_bitstring[u] == best_bitstring[v]]

    nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=cut_edges,
                           edge_color='red', width=3, style='solid')
    nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=non_cut_edges,
                           edge_color='gray', width=1.5, style='dashed')

    axes[0].set_title(f'Best Partition Found\nCut = {sum(1 for _ in cut_edges)} edges',
                      fontsize=13, fontweight='bold')
    axes[0].axis('off')

    blue_patch = mpatches.Patch(color='steelblue', label='Group 0')
    orange_patch = mpatches.Patch(color='darkorange', label='Group 1')
    axes[0].legend(handles=[blue_patch, orange_patch], fontsize=10)

    # Right: unpartitioned graph for reference
    nx.draw_networkx(G, pos, ax=axes[1], node_color='lightgray',
                     node_size=800, font_weight='bold',
                     edge_color='gray', width=2)
    axes[1].set_title('Original Graph', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle('QAOA MaxCut — Graph Partition Visualization',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def compute_cut(bitstring, G):
    """Helper to compute cut value."""
    return sum(1 for u, v in G.edges() if bitstring[u] != bitstring[v])

def plot_p_layers(p_values, approx_ratios, expected_cuts, save_path):
    """Plots p layers vs approximation ratio and expected cut."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = 'steelblue'
    ax1.bar(p_values, approx_ratios, color=color1, alpha=0.7,
            label='Approximation Ratio', zorder=3)
    ax1.set_xlabel('QAOA Layers (p)', fontsize=13)
    ax1.set_ylabel('Approximation Ratio', fontsize=13, color=color1)
    ax1.set_ylim(0, 1.2)
    ax1.axhline(y=1.0, color='green', linestyle='--',
                linewidth=1.8, label='Optimal ratio = 1.0')
    ax1.set_xticks(p_values)

    # Annotate bars
    for p, ratio in zip(p_values, approx_ratios):
        ax1.text(p, ratio + 0.02, f'{ratio:.3f}',
                 ha='center', fontsize=10, fontweight='bold', color=color1)

    ax2 = ax1.twinx()
    color2 = 'darkorange'
    ax2.plot(p_values, expected_cuts, marker='o', color=color2,
             linewidth=2.5, markersize=8, label='Expected Cut Value')
    ax2.set_ylabel('Expected Cut Value', fontsize=13, color=color2)
    ax2.set_ylim(2.5, 4.5)

    # Annotate expected cuts
    for p, ec in zip(p_values, expected_cuts):
        ax2.text(p + 0.08, ec + 0.05, f'{ec:.2f}',
                 fontsize=9, color=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='lower right')

    ax1.set_title('QAOA Layers (p) vs Solution Quality', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")
def save_results(data_dict, filepath):
    """Saves results to CSV."""
    df = pd.DataFrame(data_dict)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")