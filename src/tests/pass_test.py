"""Verifies that a given circuit is equivalent following transformation passes.

In the case of mirror-gate transformations, the output order of qubits
may be changed. However, each transformation should have an optional
parameter, `preserve_ordering`, such that additional SWAP gates are
included to ensure the output of qubits is the same as the input.
"""

# import pytest
# from qiskit.circuit.random import random_circuit
# from qiskit.quantum_info import Operator
# from qiskit.transpiler import PassManager
# from qiskit.transpiler.coupling import CouplingMap

# # from virtual_swap.deprecated.cns_sabre_v2 import CNS_SabreSwap_V2
# from mirror_gates.mirage import Mirage


# TODO
def test_foo():
    """Test nothing."""
    assert True


# # Function to build circuits.
# def build_circuits():
#     """Build a list of random quantum circuits for testing.

#     Modify this function to build your specific circuits.
#     """
#     circuits = []
#     for i in range(1, 3):
#         circuits.append(random_circuit(num_qubits=i, depth=i, max_operands=2))
#     return circuits


# # Function to define your set of transformation passes.
# def transformation_passes():
#     """Define a list of transformation passes for testing."""
#     coupling_map = CouplingMap.from_grid(2, 2)
#     passes = [Mirage(coupling_map)]
#     return passes


# # Generate all pairs of circuits and passes
# test_cases = []
# for circuit in build_circuits():
#     for transformation_pass in transformation_passes():
#         test_cases.append((circuit, transformation_pass))

# # Generate test case ids
# test_case_ids = []
# for i, transformation_pass in enumerate(transformation_passes()):
#     for j in range(len(build_circuits())):
#         test_case_ids.append(f"Circuit_{j+1}_{type(transformation_pass).__name__}")


# @pytest.mark.parametrize("test_case", test_cases, ids=test_case_ids)
# def test_transformation_pass(test_case):
#     """Test case for a transformation pass.

#     The test passes if the circuit is equivalent before and after the transformation,
#     checked using Operator.equiv().
#     """
#     circuit, transformation_pass = test_case

#     # Create a pass manager for the specific transformation pass
#     pass_manager = PassManager()
#     pass_manager.append(transformation_pass)

#     # Apply the transformation
#     transformed_circuit = pass_manager.run(circuit)

#     # Check circuit is equivalent to itself before and after the transformation.
#     assert Operator(circuit).equiv(
#         Operator(transformed_circuit)
#     ), f"{type(transformation_pass).__name__} broke equivalence"
