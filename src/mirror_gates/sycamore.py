"""Define the Sycamore gate and its decompositions."""
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import CXGate, CZGate, SwapGate, iSwapGate
from qiskit.quantum_info import Operator

qc = QuantumCircuit(2)
qc.append(iSwapGate().power(-1), [0, 1])
qc.append(CZGate().power(1 / 6), [0, 1])
syc = Operator(qc)
syc.name = "syc"
syc.params = []


cx_decomp = QuantumCircuit.from_qasm_file("../circuits/d02_syc_cnot.qasm")
sel.add_equivalence(CXGate(), cx_decomp)

swap_decomp = QuantumCircuit.from_qasm_file("../circuits/d02_syc_swap.qasm")
sel.add_equivalence(SwapGate(), swap_decomp)
