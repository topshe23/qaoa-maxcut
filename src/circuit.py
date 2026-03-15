# src/circuit.py
# Builds the QAOA circuit for MaxCut problem

from qiskit.circuit import QuantumCircuit, ParameterVector
import numpy as np
import networkx as nx


def build_graph(n_nodes=4, edge_list=None):
    """
    Builds the graph for MaxCut.
    Default: 4-node cycle graph — simple but non-trivial.

    Args:
        n_nodes: number of nodes
        edge_list: list of (node1, node2) tuples
    Returns:
        G: networkx Graph object
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    if edge_list is None:
        # Default: 4-node graph with 5 edges
        # Optimal MaxCut = 4 (verified by brute force)
        edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
    else:
        edges = edge_list

    G.add_edges_from(edges)
    return G


def build_qaoa_circuit(G, p=1):
    """
    Builds the QAOA circuit for MaxCut on graph G.

    Structure:
        |+⟩^n                    ← equal superposition
        → Cost layer (γ)         ← encodes cut edges
        → Mixer layer (β)        ← explores solutions
        → repeat p times

    Args:
        G: networkx Graph
        p: number of QAOA layers (more = better approximation)
    Returns:
        qc: parameterized QuantumCircuit
        gamma: ParameterVector for cost angles
        beta: ParameterVector for mixer angles
    """
    n = G.number_of_nodes()

    # 2p parameters total: p gamma values + p beta values
    gamma = ParameterVector('γ', p)
    beta  = ParameterVector('β', p)

    qc = QuantumCircuit(n)

    # Step 1: Initialize all qubits in superposition |+⟩
    # This represents trying ALL possible cuts simultaneously
    for i in range(n):
        qc.h(i)
    qc.barrier(label='Init |+⟩')

    # Step 2: p layers of cost + mixer
    for layer in range(p):

        # --- Cost layer (Problem Unitary) ---
        # For each edge (u,v): apply ZZ interaction
        # This rewards states where u and v are in different groups
        qc.barrier(label=f'Cost γ[{layer}]')
        for u, v in G.edges():
            qc.cx(u, v)
            qc.rz(2 * gamma[layer], v)
            qc.cx(u, v)

        # --- Mixer layer (Mixing Unitary) ---
        # Applies RX rotation to each qubit
        # This allows exploring different cut configurations
        qc.barrier(label=f'Mixer β[{layer}]')
        for i in range(n):
            qc.rx(2 * beta[layer], i)

    # Measure all qubits
    qc.measure_all()

    return qc, gamma, beta


def draw_circuit(qc, save_path='images/circuit.png'):
    """Saves circuit diagram."""
    fig = qc.draw(output='mpl', fold=-1)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Circuit saved to {save_path}")


def compute_maxcut_value(bitstring, G):
    """
    Computes the cut value for a given bitstring.
    Bitstring assigns each node to group 0 or group 1.
    Cut value = number of edges between the two groups.

    Args:
        bitstring: string like '0101' 
        G: networkx Graph
    Returns:
        cut_value: int
    """
    cut = 0
    for u, v in G.edges():
        if bitstring[u] != bitstring[v]:
            cut += 1
    return cut


def brute_force_maxcut(G):
    """
    Finds exact MaxCut by trying all 2^n partitions.
    Only feasible for small graphs.
    Returns best cut value and best partition.
    """
    n = G.number_of_nodes()
    best_cut = 0
    best_partition = None

    for i in range(2**n):
        bitstring = format(i, f'0{n}b')
        cut = compute_maxcut_value(bitstring, G)
        if cut > best_cut:
            best_cut = cut
            best_partition = bitstring

    return best_cut, best_partition


if __name__ == "__main__":
    # Build graph and circuit
    G = build_graph()
    qc, gamma, beta = build_qaoa_circuit(G, p=1)

    print("Graph:")
    print(f"  Nodes: {list(G.nodes())}")
    print(f"  Edges: {list(G.edges())}")

    best_cut, best_partition = brute_force_maxcut(G)
    print(f"\nOptimal MaxCut:")
    print(f"  Cut value: {best_cut}")
    print(f"  Partition: {best_partition}")

    print(f"\nQAOA Circuit:")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Parameters: {qc.num_parameters}")
    print(f"  Depth: {qc.depth()}")

    draw_circuit(qc)
    print(qc.draw(output='text'))