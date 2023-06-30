"""Utilities for transpiler passes."""

import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from transpile_benchy.metrics.abc_metrics import DoNothing, MetricInterface
from transpile_benchy.passmanagers.abc_runner import CustomPassManager


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
