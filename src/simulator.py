# src/simulator.py
# Runs QAOA circuit and extracts MaxCut results

import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from scipy.optimize import minimize
from src.circuit import build_graph, build_qaoa_circuit, compute_maxcut_value


def run_circuit(circuit, parameter_values, shots=1024):
    """
    Binds parameters and runs the QAOA circuit.
    Returns measurement counts dict.
    """
    # Bind all parameters
    param_dict = dict(zip(circuit.parameters, parameter_values))
    bound_circuit = circuit.assign_parameters(param_dict)

    sim = AerSimulator()
    compiled = transpile(bound_circuit, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts


def compute_expected_cut(counts, G):
    """
    Computes expected cut value from measurement counts.
    This is the cost function QAOA minimizes (we negate it
    because scipy minimizes, but we want to maximize cut).

    Args:
        counts: dict of bitstring -> count
        G: networkx Graph
    Returns:
        expected_cut: float (weighted average cut value)
    """
    total = sum(counts.values())
    expected_cut = 0
    for bitstring, count in counts.items():
        # Qiskit returns bitstrings in reverse order
        bitstring_reversed = bitstring[::-1]
        cut = compute_maxcut_value(bitstring_reversed, G)
        expected_cut += cut * (count / total)
    return expected_cut


def run_qaoa(G, p=1, shots=2048, max_iter=100):
    """
    Runs full QAOA optimization loop.
    Uses COBYLA to maximize expected cut value.

    Args:
        G: networkx Graph
        p: QAOA layers
        shots: measurement shots per evaluation
        max_iter: max optimizer iterations
    Returns:
        result dict with optimal params, best bitstring, cut value, history
    """
    qc, gamma, beta = build_qaoa_circuit(G, p=p)

    cut_history = []
    iteration = [0]

    def cost_function(params):
        counts = run_circuit(qc, params, shots=shots)
        expected_cut = compute_expected_cut(counts, G)
        cut_history.append(expected_cut)
        iteration[0] += 1
        if iteration[0] % 10 == 0:
            print(f"  Iteration {iteration[0]}: expected cut = {expected_cut:.4f}")
        # Negate because scipy minimizes
        return -expected_cut

    # Random initial parameters in [0, pi]
    np.random.seed(42)
    initial_params = np.random.uniform(0, np.pi, 2 * p)

    print(f"Starting QAOA (p={p}) with {2*p} parameters...")
    result = minimize(
        cost_function,
        initial_params,
        method='COBYLA',
        options={'maxiter': max_iter, 'rhobeg': 0.5}
    )

    # Get best solution from final parameters
    final_counts = run_circuit(qc, result.x, shots=8192)
    best_bitstring, best_cut = get_best_solution(final_counts, G)

    return {
        'optimal_params': result.x,
        'best_bitstring': best_bitstring,
        'best_cut': best_cut,
        'expected_cut': -result.fun,
        'cut_history': cut_history,
        'iterations': len(cut_history),
        'counts': final_counts
    }


def get_best_solution(counts, G):
    """
    Extracts the bitstring with highest cut value from counts.
    Returns best bitstring and its cut value.
    """
    best_cut = 0
    best_bitstring = None

    for bitstring in counts:
        bitstring_reversed = bitstring[::-1]
        cut = compute_maxcut_value(bitstring_reversed, G)
        if cut > best_cut:
            best_cut = cut
            best_bitstring = bitstring_reversed

    return best_bitstring, best_cut