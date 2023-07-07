"""Utilities for transpiler passes."""

import numpy as np
from monodromy.coordinates import unitary_to_monodromy_coordinate
from qiskit.circuit import Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from qiskit.transpiler.passes import Collect2qBlocks
from transpile_benchy.metrics.abc_metrics import DoNothing, MetricInterface
from transpile_benchy.passmanagers.abc_runner import CustomPassManager

# fast construction UnitaryGate
swap_matrix = SwapGate().to_matrix()


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

        # TODO cached_property?
        self._monodromy_coord = None


class FastConsolidateBlocks(TransformationPass):
    """Fast ConsolidateBlocks pass.

    Annotates DAGOpNodes with monodromy coordinates with caching. While passing through
    operators here, we have the ability to save time, because following consolidation,
    looking up OpNodes would be cache misses if differ by exterior 1Q gates.
    """

    # Class variable shared across all instances
    monodromy_cache = {}

    # XXX: will unitary matrices differ with circuit global phase?
    # I'm not sure that they will, but I'll leave this here for now
    @staticmethod
    def unitary_to_coordinate(unitary: UnitaryGate):
        """Convert a unitary to a monodromy coordinate."""
        # XXX is this enough precision?
        unitary_key = unitary.to_matrix().round(decimals=6).tobytes()
        if unitary_key in FastConsolidateBlocks.monodromy_cache:
            return FastConsolidateBlocks.monodromy_cache[unitary_key]
        else:
            monodromy_coord = unitary_to_monodromy_coordinate(unitary.to_matrix())
            FastConsolidateBlocks.monodromy_cache[unitary_key] = monodromy_coord
            return monodromy_coord

    def __init__(self, coord_caching=True):
        """Initialize the pass.

        Args:
            coord_caching (bool): whether to cache monodromy coordinates
            If True, will compute unitaries with ...
        """
        super().__init__()
        self.requires = [Collect2qBlocks()]
        self.coord_caching = coord_caching

    def _block_qargs_to_indices(self, block_qargs):
        """Map each qubit in block_qargs to its wire among the block's wires."""
        block_indices = [self.global_index_map[q] for q in block_qargs]
        ordered_block_indices = {
            bit: index for index, bit in enumerate(sorted(block_indices))
        }
        block_positions = {
            q: ordered_block_indices[self.global_index_map[q]] for q in block_qargs
        }
        return block_positions

    def _format_block(self, block):
        """Partition a block into front, interior, and back.

        This function returns the front, interior_block, and back gates based on the
        first and last 2Q gate in the block.
        """
        two_qubit_indices = [i for i, gate in enumerate(block) if len(gate.qargs) == 2]

        if two_qubit_indices:
            first_two_qubit_index = two_qubit_indices[0]
            last_two_qubit_index = two_qubit_indices[-1]
        else:
            # If there are no 2-qubit gates, all gates are 1-qubit
            return block, [], []

        # All gates before the first 2-qubit gate go to front
        front = block[:first_two_qubit_index]

        # All gates after the last 2-qubit gate go to back
        back = block[last_two_qubit_index + 1 :]

        # All gates between the first and last 2-qubit gate go to the interior block
        interior_block = block[first_two_qubit_index : last_two_qubit_index + 1]

        return front, interior_block, back

    def _compute_unitary_delay_exterior_1q(self, block, qubit_map) -> UnitaryGate:
        """Compute the unitary of a block of gates, optimized for coordinate caching."""
        # assuming 'exterior' 1-qubit gates are the first and last gates
        front, interior_block, back = self._format_block(block)

        # compute unitary of interior block
        interior = self._compute_unitary(interior_block, qubit_map)
        interior_key = interior.to_matrix().tobytes()

        # check if interior unitary is in cache
        if interior_key in self.monodromy_cache:
            _monodromy_coord = self.monodromy_cache[interior_key]
        else:
            # compute monodromy coordinate
            _monodromy_coord = FastConsolidateBlocks.unitary_to_coordinate(interior)
            # update cache
            self.monodromy_cache[interior_key] = _monodromy_coord

        # compute 'exterior' 1-qubit gates
        front_op = self._compute_unitary(front, qubit_map)
        back_op = self._compute_unitary(back, qubit_map)

        operator = back_op.to_matrix() @ interior.to_matrix() @ front_op.to_matrix()
        unitary = NoCheckUnitary(operator)
        unitary._monodromy_coord = _monodromy_coord

        return unitary

    def _compute_unitary(self, block, qubit_map) -> UnitaryGate:
        """Compute the unitary of a block of gates."""
        operator = np.eye(4)  # 2-qubit operator
        for gate in block:
            try:
                gate_operator = gate.op.to_matrix()
            except CircuitError:
                gate_operator = Operator(gate.op).data

            if len(gate.qargs) == 1:  # 1-qubit gate
                idx = qubit_map[gate.qargs[0]]
                # XXX careful of endianness here
                if idx == 0:  # 1-qubit gate acts on first qubit
                    gate_operator = np.kron(np.eye(2), gate_operator)
                else:  # 1-qubit gate acts on second qubit
                    gate_operator = np.kron(gate_operator, np.eye(2))

            else:  # 2-qubit gate
                # The qubit order must match the order in which the gate was applied
                if qubit_map[gate.qargs[0]] == 1:
                    # The qubit order is reversed
                    def swap_rows_columns_list(unitary):
                        """Handle swapping qarg ordering."""
                        unitary[:, [1, 2]] = unitary[:, [2, 1]]
                        unitary[[1, 2], :] = unitary[[2, 1], :]
                        return unitary

                    gate_operator = swap_rows_columns_list(gate_operator)
            operator = gate_operator @ operator
        return NoCheckUnitary(operator)

    def run(self, dag):
        """Run the ConsolidateBlocks pass on `dag`.

        Iterate over each run and replace it with an equivalent NoCheckUnitary
        on the same wires.
        """
        assert "block_list" in self.property_set
        blocks = self.property_set["block_list"]

        # Compute ordered indices for the global circuit wires
        self.global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}

        for block in blocks:
            involved_qubits = set(qarg for gate in block for qarg in gate.qargs)
            assert len(involved_qubits) == 2, "runs should be across 2 qubits"

            qubit_map = self._block_qargs_to_indices(involved_qubits)

            if self.coord_caching:
                unitary = self._compute_unitary_delay_exterior_1q(block, qubit_map)
            else:
                unitary = self._compute_unitary(block, qubit_map)

            dag.replace_block_with_op(block, unitary, qubit_map, cycle_check=False)

        del self.property_set["block_list"]

        return dag


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

    def run(self, dag):
        """Run the pass."""
        # for every parameter, assign a random value [0, 2pi]
        # not sure I good way to do this, do messy in meantime
        qc = dag_to_circuit(dag)
        for param in qc.parameters:
            qc.assign_parameters({param: np.random.uniform(0, 2 * np.pi)}, inplace=True)
        return circuit_to_dag(qc)


class RemoveAllMeasurements(TransformationPass):
    """Remove all measurements from the circuit."""

    def run(self, dag):
        """Run the pass."""
        dag.remove_all_ops_named("measure")
        return dag


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
