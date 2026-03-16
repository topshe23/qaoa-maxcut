import numpy as np
import sys
sys.path.insert(0, '.')
from src.circuit import build_graph, brute_force_maxcut
from src.simulator import run_qaoa

G = build_graph()
best_cut, best_partition = brute_force_maxcut(G)
print(f"Optimal MaxCut: {best_cut} (partition: {best_partition})")

result = run_qaoa(G, p=1, shots=2048, max_iter=50)
print(f"QAOA best cut: {result['best_cut']}")
print(f"QAOA best bitstring: {result['best_bitstring']}")
print(f"QAOA expected cut: {result['expected_cut']:.4f}")
print(f"Approximation ratio: {result['best_cut'] / best_cut:.4f}")