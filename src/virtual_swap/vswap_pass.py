"""Class for virtual-swap routing.

Use :class:qiskit.transpiler.basepasses.TransformationPass.
The idea of virtual-swap is a swap gate that is performed by logical-
physical mapping. Rather than performing a SWAP, the virtual-swap,
vSWAP, relabels the logical qubits, and in effect can be thought of as
SWAP.
"""
import random
from typing import Dict

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler import TransformationPass


class VirtualSwapAnnealing(TransformationPass):
    """Virtual-swap routing."""

    def __init__(self, coupling_map, seed=None):
        """Virtual-swap routing initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            seed (int): Random seed for the stochastic part of the algorithm.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.seed = seed
        random.seed(self.seed)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the VirtualSwapAnnealing pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.
        """
        layout_dict = {qubit: qubit for qubit in dag.qubits}

        # Simulated annealing loop
        for i in range(100):
            # Make a deep copy of the layout and DAGCircuit
            current_layout_dict = layout_dict.copy()
            current_dag = dag._deep_copy()

            # Choose a random 2-qubit gate in the DAG
            node = self._random_2q_gate(dag)

            # Apply virtual-swap at the chosen gate
            self._apply_virtual_swap(current_dag, node, current_layout_dict)

            # Calculate cost of the current and new mapping
            current_cost = self._cost(dag, layout_dict)
            new_cost = self._cost(current_dag, current_layout_dict)

            # Decide whether to accept the new mapping
            if self._accept(new_cost, current_cost):
                # Update the current mapping and DAGCircuit
                layout_dict = current_layout_dict
                dag = current_dag

        return dag

    def _random_2q_gate(self, dag: DAGCircuit) -> DAGNode:
        """Choose a random 2-qubit gate in the DAG."""
        two_qubit_nodes = [
            node for node in dag.nodes() if node.type == "op" and len(node.qargs) == 2
        ]
        return random.choice(two_qubit_nodes)

    def _cost(self, dag: DAGCircuit, layout_dict: Dict[Qubit, Qubit]) -> float:
        """Calculate the cost of the current mapping."""
        critical_path_length = dag.depth()
        return critical_path_length

    def _accept(self, new_cost: float, current_cost: float) -> bool:
        """Decide whether to accept the new mapping."""
        if new_cost <= current_cost:
            return True
        else:
            probability = min(1, current_cost / new_cost)
            return random.random() < probability

    def _apply_virtual_swap(
        self, dag: DAGCircuit, node: DAGNode, layout_dict: Dict[Qubit, Qubit]
    ) -> None:
        """Apply a virtual-swap.

        Apply a virtual-swap at the given node in the DAG and update the layout.

        Args:
            dag (DAGCircuit): DAG to map.
            node (DAGNode): Node at which to apply the virtual-swap.
            layout_dict (Dict[Qubit, Qubit]): Current layout of qubits.
        """
        if node.type != "op" or len(node.qargs) != 2:
            return

        # Update the layout dictionary
        layout_dict[node.qargs[0]], layout_dict[node.qargs[1]] = (
            layout_dict[node.qargs[1]],
            layout_dict[node.qargs[0]],
        )

        # Propagate the changes through the remaining gates in the DAG
        for successor_node in dag.successors(node):
            if successor_node.type == "op":
                new_qargs = [layout_dict[q] for q in successor_node.qargs]
                successor_node.qargs = new_qargs
