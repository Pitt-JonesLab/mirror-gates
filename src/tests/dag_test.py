from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator

from virtual_swap.cns_transform import _cns_transform


def generate_circuit():
    qc = QuantumCircuit(3)
    qc.t(2)
    qc.iswap(0, 2)
    qc.s(0)
    qc.h(1)
    qc.swap(0, 1)
    qc.h(1)
    qc.cx(1, 2)
    qc.t(1)
    qc.rzz(0.5, 2, 0)
    qc.cx(0, 2)
    qc.h(1)
    qc.h(2)
    return qc


def test_single_node_cns_transform():
    # Generate the original circuit and calculate its operator
    qc_original = generate_circuit()
    dag_original = circuit_to_dag(qc_original)
    op_original = Operator(qc_original)

    # Identify the node to be transformed
    node_to_transform = dag_original.two_qubit_ops()[0]

    # Apply the CNS transformation
    dag_transformed = _cns_transform(
        dag_original, node_to_transform, preserve_layout=True
    )
    qc_transformed = dag_to_circuit(dag_transformed)

    # Test equivalence of original and transformed circuits
    assert op_original.equiv(Operator(qc_transformed))


def test_multiple_nodes_cns_transform():
    # Generate the original circuit and calculate its operator
    qc_original = generate_circuit()
    dag_original = circuit_to_dag(qc_original)
    op_original = Operator(qc_original)

    # Identify the nodes to be transformed
    # should test the function works if passed neither a CX nor an iSWAP
    nodes_to_transform = dag_original.op_nodes()

    # Apply the CNS transformation
    dag_transformed = _cns_transform(
        dag_original, *nodes_to_transform, preserve_layout=True
    )
    qc_transformed = dag_to_circuit(dag_transformed)

    # Test equivalence of original and transformed circuits
    assert op_original.equiv(Operator(qc_transformed))


if __name__ == "__main__":
    test_single_node_cns_transform()
    test_multiple_nodes_cns_transform()
