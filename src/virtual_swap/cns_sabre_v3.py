"""V3 Rewrite of CNS SABRE Implementation.

Goal is to remove the spaghetti code we created when modifying the original,
more importantly, I believe I need to decouple the routing from the decomposition
for the sake of unit testing. We should be able to keep the input and output qubits
the same, and can verify the correctness of the CNS sub by comparing the output
of this pass to the original circuit. Also, I hope to improve general readability
and modularity.


Design notes:
- wait until end to process swaps on the return dag,
this lets us unit test the CNS subs without worrying about data movement changes to the layout
- eval local cost changes between intermediate layer -> front layer
    - but can define a brute strat or sequential strat
    - also could investigate look ahead
- enforce a consolidate prereq, this means we have to use coordinates
    - can assume circuit has no 1Q gates, makes much simpler
    - fixes unreliable look up on iswap, cx gates
    - future: a more advanced method should consider arbitrary basis
    - easier to reason about costs, since already consolidated
- use existing methods from parent class when possible
- keep the run() simple and modular,
- can use my own formatting and docuemntation style
- handle all processing at the end, means can use cns_transform instead of _get_node_cns
"""

import random
from itertools import combinations

from virtual_swap.cns_transform import _get_node_cns

# NOTE, the current qiskit version uses a rust backend
# we have in deprecated/ the original python implementation we modified in V2.
# might be useful to grab some functions from it
from virtual_swap.deprecated.sabre_swap import SabreSwap as LegacySabreSwap


class CNS_Sabre(LegacySabreSwap):
    def __init__(self, coupling_map, heuristic="lookahead"):
        super().__init__(coupling_map, heuristic=heuristic)

    def run(self, dag):
        # Initialize variables
        self._initialize_variables(dag)

        # Start algorithm from the front layer and iterate until all gates done.
        self.required_predecessors = self._build_required_predecessors(dag)

        # Main loop for processing front layer
        while self.front_layer:
            self._process_front_layer(dag)

            # If no gates can be executed, add greedy swaps
            if (
                not self.execute_gate_list
                and len(self.ops_since_progress) > self.max_iterations_without_progress
            ):
                self._handle_no_progress(dag)

            # If there are gates to be executed, process them
            if self.execute_gate_list:
                self._process_execute_gate_list(dag)

            # If there are gates in the intermediate layer, process them
            if self.intermediate_layer:
                self._process_intermediate_layer_with_permutations(dag)

            # After all free gates are exhausted, find and insert the best swap
            if not self.extended_set:
                self._find_and_insert_best_swap(dag)

        # Process remaining gates in the intermediate layer
        self._process_remaining_intermediate_layer(dag)

        # Set final layout and return the mapped dag
        self.property_set["final_layout"] = self.current_layout
        return self.mapped_dag

    def _process_intermediate_layer_with_permutations(self, dag):
        # Iterate over all permutations of CNS substitutions
        for r in range(len(self.cns_sub_candidates) + 1):
            combs = list(combinations(self.cns_sub_candidates, r))

            # Randomly select combinations if there are more than max_combinations
            if len(combs) > self.max_combinations:
                combs = random.sample(combs, self.max_combinations)

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
