# src/experiment.py
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.circuit import build_graph, brute_force_maxcut, build_qaoa_circuit
from src.simulator import run_qaoa, run_circuit, compute_expected_cut, get_best_solution
from src.utils import (plot_convergence, plot_measurement_distribution,
                       plot_graph_partition, save_results, compute_cut)

def experiment_qaoa_convergence():
    """
    Experiment 1: Run QAOA p=1 and track convergence.
    Shows expected cut rising toward optimal over iterations.
    """
    print("\nExperiment 1: QAOA Convergence (p=1)...")
    G = build_graph()
    best_cut, best_partition = brute_force_maxcut(G)

    result = run_qaoa(G, p=1, shots=2048, max_iter=100)

    print(f"  Optimal MaxCut:     {best_cut}")
    print(f"  QAOA best cut:      {result['best_cut']}")
    print(f"  Expected cut:       {result['expected_cut']:.4f}")
    print(f"  Approximation ratio: {result['best_cut']/best_cut:.4f}")
    print(f"  Best partition:     {result['best_bitstring']}")

    return result, G, best_cut


def experiment_p_layers():
    """
    Experiment 2: How does number of QAOA layers (p) affect quality?
    More layers = better approximation ratio.
    """
    print("\nExperiment 2: p-layers vs Approximation Ratio...")
    G = build_graph()
    best_cut, _ = brute_force_maxcut(G)

    p_values = [1, 2, 3, 4]
    approx_ratios = []
    expected_cuts = []

    for p in p_values:
        result = run_qaoa(G, p=p, shots=2048, max_iter=150)
        ratio = result['best_cut'] / best_cut
        approx_ratios.append(ratio)
        expected_cuts.append(result['expected_cut'])
        print(f"  p={p}: best cut={result['best_cut']}, "
              f"expected={result['expected_cut']:.4f}, "
              f"ratio={ratio:.4f}")

    return p_values, approx_ratios, expected_cuts


def experiment_shots_vs_accuracy():
    """
    Experiment 3: How does shot count affect solution quality?
    More shots = more reliable probability estimates.
    """
    print("\nExperiment 3: Shots vs Solution Accuracy...")
    G = build_graph()
    best_cut, _ = brute_force_maxcut(G)

    # First get optimal parameters from a good run
    result_opt = run_qaoa(G, p=1, shots=4096, max_iter=100)
    optimal_params = result_opt['optimal_params']
    qc, _, _ = build_qaoa_circuit(G, p=1)

    shot_counts = [64, 128, 256, 512, 1024, 2048, 4096]
    expected_cuts = []
    best_cuts = []

    for shots in shot_counts:
        counts = run_circuit(qc, optimal_params, shots=shots)
        expected_cut = compute_expected_cut(counts, G)
        best_bitstring, best_cut_val = get_best_solution(counts, G)
        expected_cuts.append(expected_cut)
        best_cuts.append(best_cut_val)
        print(f"  shots={shots}: expected cut={expected_cut:.4f}, "
              f"best cut={best_cut_val}")

    return shot_counts, expected_cuts, best_cuts


if __name__ == "__main__":

    # --- Experiment 1: Convergence ---
    conv_result, G, optimal_cut = experiment_qaoa_convergence()

    plot_convergence(
        conv_result['cut_history'],
        conv_result['expected_cut'],
        optimal_cut,
        'results/plots/qaoa_convergence.png'
    )
    plot_measurement_distribution(
        conv_result['counts'], G,
        conv_result['best_bitstring'],
        'results/plots/measurement_distribution.png'
    )
    plot_graph_partition(
        G, conv_result['best_bitstring'],
        'results/plots/graph_partition.png'
    )
    save_results(
        {'iteration': list(range(len(conv_result['cut_history']))),
         'expected_cut': conv_result['cut_history']},
        'results/data/convergence_results.csv'
    )

    # --- Experiment 2: p layers ---
    p_values, approx_ratios, expected_cuts_p = experiment_p_layers()

    save_results(
        {'p': p_values, 'approx_ratio': approx_ratios,
         'expected_cut': expected_cuts_p},
        'results/data/p_layers_results.csv'
    )

    # --- Experiment 3: Shots ---
    shot_counts, expected_cuts_s, best_cuts_s = experiment_shots_vs_accuracy()

    save_results(
        {'shots': shot_counts, 'expected_cut': expected_cuts_s,
         'best_cut': best_cuts_s},
        'results/data/shots_results.csv'
    )

    print("\nAll experiments done!")
    print("Plots saved to results/plots/")
    print("CSVs saved to results/data/")