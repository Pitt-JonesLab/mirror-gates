"""Define the Sycamore gate and its decompositions."""
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, iSwapGate
from qiskit.quantum_info import Operator

qc = QuantumCircuit(2)
qc.append(iSwapGate().power(-1), [0, 1])
qc.append(CZGate().power(1 / 6), [0, 1])
syc = Operator(qc)
syc.name = "syc"
syc.params = []

# # Get the directory of the current file
# current_directory = Path(__file__).parent

# # Compute the absolute paths to the .qasm files
# cx_decomp_path = current_directory / "../circuits/d02_syc_cnot.qasm"
# swap_decomp_path = current_directory / "../circuits/d02_syc_swap.qasm"

# cx_decomp = QuantumCircuit.from_qasm_file(cx_decomp_path)
# cx_decomp = cx_decomp.decompose()
# sel.add_equivalence(CXGate(), cx_decomp)

# swap_decomp = QuantumCircuit.from_qasm_file(swap_decomp_path)
# sel.add_equivalence(SwapGate(), swap_decomp)
