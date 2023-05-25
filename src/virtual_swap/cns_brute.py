"""Brute force CNS substitution pass.

This pass explores the potential benefits of treating CNOT and CNOT+SWAP
(CNS) operations as equivalent during the transpilation process, based
on the principle that both decompose into two iswap gates. Thus, the
SWAP operation in a CNOT+SWAP sequence can be seen as a "free"
operation, as it does not contribute to the iswap gate count. This
allows for additional data movement in the circuit without increasing
the gate count. We leverage this insight by iterating over all possible
permutations of CNOT and CNOT+SWAP substitutions within a given circuit,
creating a new quantum circuit for each permutation. Each new circuit is
then passed through layout and routing stages, and the most efficient
result is retained. This pass serves as a brute-force method to verify
the effectiveness of more complex heuristics that also consider gate
decomposition during the layout and routing stages. It is not intended
for large-scale use due to its high computational complexity.
"""

from itertools import permutations

import retworkx
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passes import SabreLayout, SabreSwap

from virtual_swap.cns_transform import _get_node_cns


class CNS_Brute(TransformationPass):
    """Brute force CNS substitution pass."""

    def __init__(self, coupling_map: CouplingMap, layout_pass=None, routing_pass=None):
        """CNS_Brute initializer."""
        super().__init__()
        self.coupling_map = coupling_map
        self.layout_pass = layout_pass if layout_pass else SabreLayout(coupling_map)
        self.routing_pass = routing_pass if routing_pass else SabreSwap(coupling_map)
        if self.property_set["layout"] is None:
            raise ValueError("CNS_Brute requires a layout")

    def run(self, dag):
        """Run the pass on the provided dag."""
        # Build a list of two-qubit gate candidates for CNS substitution
        cns_sub_candidates = [
            node for node in dag.topological_op_nodes() if node.name in ["cx", "iswap"]
        ]

        # Initialize the best_depth to the total cost of the original dag
        best_depth = self.calculate_gate_cost(dag)

        # Initialize best_dag to the original dag
        best_dag = dag

        # Iterate over all permutations of CNS substitutions
        for perm in permutations(cns_sub_candidates):
            # Create a copy of the original dag structure without operations (nodes)
            trial_dag = dag.copy_empty_like()

            # Create a copy of the layout
            trial_layout = self.property_set["layout"].copy()

            # Iterate over all nodes in topological order
            for node in dag.topological_op_nodes():
                # Map qargs to the new layout
                qargs_prime = [trial_layout[qarg] for qarg in node.qargs]

                # If the node is in the permutation list, replace with its CNS sub
                if node in perm:
                    trial_dag.apply_operation_back(_get_node_cns(node).op, qargs_prime)
                    trial_layout.swap(node.qargs[0], node.qargs[1])
                else:
                    trial_dag.apply_operation_back(node.op, qargs_prime)

            # Run the layout pass on the trial dag
            trial_dag = self.layout_pass.run(trial_dag)

            # Update the property set of the routing pass and run it on the trial dag
            self.routing_pass.property_set = self.layout_pass.property_set
            trial_dag = self.routing_pass.run(trial_dag)

            # Update the best_depth if the cost of the trial dag is smaller
            trial_depth = self.calculate_gate_cost(trial_dag)
            if trial_depth < best_depth:
                best_depth = trial_depth
                best_dag = trial_dag

        # Return the best dag found
        return best_dag

    def calculate_gate_cost(self, dag):
        """Calculate critical path 2Q gate cost."""

        def weight_fn(_1, node, _2):
            """Weight function for the longest path algorithm."""
            target_node = dag._multi_graph[node]
            if not isinstance(target_node, DAGOpNode) or len(target_node.qargs) != 2:
                return 0
            elif target_node.name in ["iswap", "cx", "iswap_prime", "cx_prime"]:
                return 2
            elif target_node.name == "swap":
                return 3
            else:
                raise ValueError("Unknown node type")

        # Compute the longest path using the weight function
        return retworkx.dag_longest_path(dag._multi_graph, weight_fn=weight_fn)
