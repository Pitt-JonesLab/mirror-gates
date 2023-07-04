"""Utilities for transpiler passes."""

import numpy as np
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions import UnitaryGate
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from transpile_benchy.metrics.abc_metrics import DoNothing, MetricInterface
from transpile_benchy.passmanagers.abc_runner import CustomPassManager

# fast construction UnitaryGate


class NoCheckUnitary(UnitaryGate, Gate):
    """Class quantum gates specified by a unitary matrix.

    Building a UnitaryGate calls an expensive is_unitary_matrix check. This class skips
    that check if we are manually constructing the gate, and that we already know it is
    unitary.
    """

    def __init__(self, data: np.ndarray, label=None):
        """Create a gate from a numeric unitary matrix.

        Args:
            data (np.ndarray): unitary operator.
        """
        assert isinstance(data, np.ndarray)
        input_dim, output_dim = data.shape
        num_qubits = int(np.log2(input_dim))
        # Store instruction params
        Gate.__init__(self, "unitary", num_qubits, [data], label=label)


# write a transformationpass that subs all IGates with U(0, 0, 0)
class RemoveIGates(TransformationPass):
    """Remove all IGates from the circuit."""

    def __init__(self):
        """Initialize the pass."""
        super().__init__()

    def run(self, dag):
        """Run the pass."""
        dag.remove_all_ops_named("id")
        return dag


class RemoveSwapGates(TransformationPass):
    """Remove all swap gates from the circuit.

    If encounter a SWAP, update gates on swapped qubits. I am not sure, but I found some
    examples by hand where our method performs badly, if given a circuit already
    containing SWAP gates.
    """

    def __init__(self):
        """Initialize the pass."""
        super().__init__()

    def no_swap_transform(self, dag: DAGCircuit):
        """Transform DAG, removes all SWAP gates."""
        new_dag = dag.copy_empty_like()

        # Initialize layout for each node
        layout = {
            qarg: qarg for node in dag.topological_op_nodes() for qarg in node.qargs
        }

        for node in dag.topological_op_nodes():
            qargs = [layout.get(qarg, qarg) for qarg in node.qargs]

            if node.op.name == "swap":
                # swap values in layout
                layout[node.qargs[0]], layout[node.qargs[1]] = qargs[1], qargs[0]
            else:
                new_dag.apply_operation_back(node.op, qargs)

        return new_dag

    def run(self, dag):
        """Run the pass."""
        return self.no_swap_transform(dag)


class SaveCircuitProgress(AnalysisPass):
    """Used to save circuit for debugging progress."""

    def __init__(self, qc_name=None):
        """Initialize the pass."""
        super().__init__()
        self.qc_name = qc_name or "circuit_progress"

    def run(self, dag):
        """Run the pass."""
        # convert dag to circuit,
        # save into property_set
        self.property_set[self.qc_name] = dag_to_circuit(dag)
        return dag


class AssignAllParameters(TransformationPass):
    """Assigns all parameters to a random value."""

    def __init__(self):
        """Initialize the pass."""
        super().__init__()

    def run(self, dag):
        """Run the pass."""
        # for every parameter, assign a random value [0, 2pi]
        # not sure I good way to do this, do messy in meantime
        qc = dag_to_circuit(dag)
        for param in qc.parameters:
            qc.assign_parameters({param: np.random.uniform(0, 2 * np.pi)}, inplace=True)
        return circuit_to_dag(qc)


class SubsMetric(MetricInterface):
    """Calculate the depth of a circuit."""

    def __init__(self):
        """Initialize the metric."""
        super().__init__(name="accepted_subs")

    def _get_pass(self, transpiler: CustomPassManager):
        """Return the pass associated with this metric.

        NOTE: this is a dummy pass, it does nothing.
        This is because the metric has been calculated in SabreMS.
        We are building a metric just so we know to save the result.
        """
        return DoNothing()
