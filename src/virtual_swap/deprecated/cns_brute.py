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

import random
from copy import deepcopy
from itertools import combinations

import numpy as np
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passes import SabreLayout, SabreSwap, Unroller

from virtual_swap.cns_transform import _get_node_cns


class CNS_Brute(TransformationPass):
    """Brute force CNS substitution pass."""

    def __init__(self, coupling_map: CouplingMap, layout_pass=None, routing_pass=None):
        """CNS_Brute initializer."""
        super().__init__()
        self.save_all = True
        self.requires.append(Unroller(basis=["u", "cx", "iswap", "swap"]))
        self.coupling_map = coupling_map
        self.layout_pass = layout_pass if layout_pass else SabreLayout(coupling_map)
        self.routing_pass = routing_pass if routing_pass else SabreSwap(coupling_map)

    def run(self, dag):
        """Run the pass on the provided dag."""
        if self.property_set["layout"] is None:
            raise ValueError("CNS_Brute requires a layout")

        self.all_dags = []
        self.all_costs = []
        self.all_perms = []

        # Build a list of two-qubit gate candidates for CNS substitution
        cns_sub_candidates = [
            node for node in dag.topological_op_nodes() if node.name in ["cx", "iswap"]
        ]

        # Initialize the best_depth to the total cost of the original dag
        best_depth = np.inf

        # Initialize best_dag to the original dag
        best_dag = dag

        # Determine the maximum number of combinations to select randomly
        max_combinations = 250

        # Iterate over all permutations of CNS substitutions
        for r in range(len(cns_sub_candidates) + 1):
            combs = list(combinations(cns_sub_candidates, r))

            # Randomly select combinations if there are more than max_combinations
            if len(combs) > max_combinations:
                combs = random.sample(combs, max_combinations)

            for perm in combs:
                # Create a copy of the original dag structure without operations (nodes)
                trial_dag = dag.copy_empty_like()

                # Create a copy of the layout
                trial_layout = self.property_set["layout"].copy()
                wire_dict = {wire: wire for wire in trial_layout.get_virtual_bits()}

                # Iterate over all nodes in topological order
                for node in dag.topological_op_nodes():
                    # Map qargs to the new layout
                    qargs_prime = [wire_dict.get(qarg, qarg) for qarg in node.qargs]

                    # If the node is in the permutation list, replace with its CNS sub
                    if node in perm:
                        trial_dag.apply_operation_back(
                            _get_node_cns(node).op, qargs_prime
                        )
                        save = wire_dict[node.qargs[0]]
                        wire_dict[node.qargs[0]] = wire_dict[node.qargs[1]]
                        wire_dict[node.qargs[1]] = save
                        # check that every value in wire_dict is unique
                        try:
                            assert len(wire_dict.values()) == len(
                                set(wire_dict.values())
                            )
                        except AssertionError:
                            print("wire_dict not unique")
                    else:
                        trial_dag.apply_operation_back(node.op, qargs_prime)

                # Run the layout pass on the trial dag
                # assert that there are 6 2Q ops in the trial dag
                try:
                    assert len(dag.two_qubit_ops()) == len(trial_dag.two_qubit_ops())
                except AssertionError:
                    print("Trial dag missing gates")

                # save a copy
                # before_routed_dag = deepcopy(trial_dag)
                trial_dag = self.layout_pass.run(trial_dag)

                # applying routing should only ever add 2Q gates (Swaps)
                try:
                    assert len(trial_dag.two_qubit_ops()) >= len(dag.two_qubit_ops())
                except AssertionError:
                    print("Routing substracted gates")

                # # Update the property set of the routing pass and run it on the trial dag
                # self.routing_pass.property_set = self.layout_pass.property_set
                # trial_dag = self.routing_pass.run(trial_dag)

                # Update the best_depth if the cost of the trial dag is smaller
                trial_depth = self.calculate_gate_cost(trial_dag)

                if self.save_all:
                    self.all_dags.append(trial_dag)
                    self.all_costs.append(trial_depth)
                    self.all_perms.append(perm)

                if trial_depth < best_depth:
                    best_depth = trial_depth
                    best_dag = trial_dag

        if self.save_all:
            # Sort all_dags, all_costs, and all_perms by all_costs
            sorted_data = sorted(
                zip(self.all_dags, self.all_costs, self.all_perms), key=lambda x: x[1]
            )
            self.all_dags, self.all_costs, self.all_perms = zip(*sorted_data)
            self.property_set.update(
                {
                    "all_dags": self.all_dags,
                    "all_costs": self.all_costs,
                    "all_perms": self.all_perms,
                }
            )

        # Return the best dag found
        return best_dag

    # def calculate_gate_cost(self, dag):
    #     """Calculate critical path 2Q gate cost."""

    #     def weight_fn(_1, node, _2):
    #         """Weight function for the longest path algorithm."""
    #         target_node = dag._multi_graph[node]
    #         if not isinstance(target_node, DAGOpNode) or len(target_node.qargs) != 2:
    #             return 0
    #         elif target_node.name in ["iswap", "cx", "iswap_prime", "cx_prime"]:
    #             return 2
    #         elif target_node.name == "swap":
    #             return 3
    #         else:
    #             raise ValueError("Unknown node type")

    #     # Compute the longest path using the weight function
    #     longest_path_length= retworkx.dag_longest_path_length(dag._multi_graph, weight_fn=weight_fn)
    #     return longest_path_length

    def calculate_gate_cost(self, dag):
        """Force into sqiswap gates then calculate critical path cost."""
        # Collect2qBlocks(),
        # ConsolidateBlocks(force_consolidate=True),
        # RootiSwapWeylDecomposition(),

        temp_dag = deepcopy(dag)
        from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
        from slam.utils.transpiler_pass.weyl_decompose import RootiSwapWeylDecomposition

        collect = Collect2qBlocks()
        consolidate = ConsolidateBlocks(force_consolidate=True)
        weyl = RootiSwapWeylDecomposition()

        temp_dag = collect.run(temp_dag)
        consolidate.property_set = collect.property_set
        temp_dag = consolidate.run(temp_dag)
        weyl.property_set = consolidate.property_set
        temp_dag = weyl.run(temp_dag)

        return len(temp_dag.two_qubit_ops())
