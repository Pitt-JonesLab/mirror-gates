"""Add custom gates to the session EquivalenceLibrary instance.

NOTE: sel is global, so just import this file before calling transpile.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library.standard_gates import CXGate, SwapGate, XXPlusYYGate

# https://arxiv.org/pdf/2110.11537.pdf

cx_decomp = QuantumCircuit(2)
cx_decomp.u(np.pi / 2, -np.pi / 2, 0, 0)
cx_decomp.u(np.pi / 2, np.pi, -np.pi / 2, 1)
cx_decomp.append(XXPlusYYGate(-np.pi / 2), [0, 1])
cx_decomp.u(np.pi, -np.pi / 2, np.pi / 2, 0)
cx_decomp.u(0, 0, np.pi, 1)
cx_decomp.append(XXPlusYYGate(-np.pi / 2), [0, 1])
cx_decomp.u(np.pi / 2, -np.pi / 2, -3 * np.pi / 2, 0)
cx_decomp.u(0, -np.pi / 2, -np.pi, 1)
sel.add_equivalence(CXGate(), cx_decomp)

swap_decomp = QuantumCircuit(2)
swap_decomp.append(XXPlusYYGate(-np.pi / 2, 0), [0, 1])
swap_decomp.rx(-np.pi / 2, 0)
swap_decomp.rx(-np.pi / 2, 1)
swap_decomp.append(XXPlusYYGate(-np.pi / 2), [0, 1])
swap_decomp.rx(np.pi / 2, 0)
swap_decomp.rx(np.pi / 2, 1)
swap_decomp.ry(-np.pi / 2, 0)
swap_decomp.ry(-np.pi / 2, 1)
swap_decomp.append(XXPlusYYGate(-np.pi / 2), [0, 1])
swap_decomp.ry(np.pi / 2, 0)
swap_decomp.ry(np.pi / 2, 1)
sel.add_equivalence(SwapGate(), swap_decomp)

# bb = transpile(
#     circ, basis_gates=["u", "xx_plus_yy"], coupling_map=topo, optimization_level=3
# )
# bb.draw(output="mpl")

#######################
# NOTE this is a hack :) #
# https://github.com/Qiskit/qiskit-terra/issues/9485
# add I = U(0, 0, 0) to the equivalence library
# from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
# from qiskit import QuantumCircuit
# from qiskit.circuit.library import IGate
# qc = QuantumCircuit(1)
# qc.u(0, 0, 0, 0)
# sel.add_equivalence(IGate(), qc)
