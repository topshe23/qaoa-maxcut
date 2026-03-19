# Experiment Summary — QAOA MaxCut

## Objective

I wanted to implement QAOA from scratch and verify it finds the optimal
MaxCut on a small graph. Three questions drove the experiments: does it
converge? does more depth actually help? and how many shots are needed
before results stabilize?

## Methodology

### Graph

A 4-node graph with 5 edges was used — edges (0,1), (0,2), (0,3),
(1,2), (2,3). Optimal MaxCut = 4, verified by brute force over all
16 possible partitions. Optimal partitions are 0101 and 1010
(equivalent under group label swapping).

### Circuit Design

The QAOA circuit has three stages:

**Initialization:** All qubits put into equal superposition with H
gates. This represents trying all 2^4 = 16 partitions simultaneously.

**Cost layer (repeated p times):** For each edge (u,v), apply
CNOT-RZ(2γ)-CNOT. This encodes the graph structure — the ZZ
interaction rewards states where u and v are in different groups.

**Mixer layer (repeated p times):** Apply RX(2β) to every qubit.
This allows the circuit to explore different partitions by rotating
between |0⟩ and |1⟩.

The optimizer tunes γ and β values (2p parameters total) to maximize
the expected cut value.

### Cost Function
```
Expected cut = sum over all bitstrings of (cut_value × probability)
```

COBYLA minimizes the negative of this (since scipy minimizes).
At each iteration the circuit is run with current parameters,
counts are measured, and expected cut is computed and returned.

## Observations

### Experiment 1 — Convergence

QAOA with p=1 converged quickly — expected cut stabilized around
3.24 within 25 iterations. The optimal bitstring 1010 was the most
frequently measured outcome, appearing ~16% of the time. The
approximation ratio reached 1.0 — meaning QAOA found the exact
optimal solution.

### Experiment 2 — p-Layers

| p | Expected Cut | Approximation Ratio |
|---|---|---|
| 1 | 3.23 | 1.00 |
| 2 | 3.84 | 1.00 |
| 3 | 3.77 | 1.00 |
| 4 | 3.17 | 1.00 |

All depths found the optimal cut. Expected cut peaked at p=2 (3.84)
then declined. This is consistent with the barren plateau problem —
deeper circuits have more parameters, making the energy landscape
flatter and harder to optimize.

### Experiment 3 — Shots

Best cut = 4 was found at every shot count from 64 to 4096. Expected
cut was noisier at low shots (variance ~0.3) but stabilized above
512 shots. For reliable expected cut estimates, 1024+ shots is
recommended.

## Key Insight

QAOA works well for this graph even at p=1. The approximation ratio
of 1.0 means it found the exact optimal solution — not just an
approximation. The more nuanced story is in the expected cut: p=2
gives the highest probability of sampling the optimal solution, while
p=1 is faster and still reliable. This is the practical tradeoff in
QAOA — depth vs optimization difficulty.

## Limitations

- Only a 4-node graph — QAOA's advantage over classical is not
  visible at this scale
- No noise model — real hardware would show degradation at higher p
- COBYLA may get stuck in local minima for larger, denser graphs
- Barren plateau effect at p=4 suggests parameter initialization
  matters more as circuit depth grows

## What I'd Do Next

- Test on larger graphs (8-12 nodes) where brute force is harder
- Add depolarizing noise and measure how approximation ratio degrades
- Compare COBYLA vs SPSA vs gradient descent convergence
- Implement warm starting — use classical greedy solution as
  initial parameters
- Run on IBM quantum hardware and compare against simulation