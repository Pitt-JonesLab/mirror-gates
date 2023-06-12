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

import numpy as np
from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

from virtual_swap.cns_transform import _get_node_cns

# NOTE, the current qiskit version uses a rust backend
# we have in deprecated/ the original python implementation we modified in V2.
# might be useful to grab some functions from it
from virtual_swap.deprecated.sabre_swap import SabreSwap as LegacySabreSwap

EXTENDED_SET_SIZE = (
    20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.
DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


class SabreVS(LegacySabreSwap):
    def __init__(self, coupling_map, heuristic="lookahead"):
        super().__init__(coupling_map, heuristic=heuristic)
        # want to force only 2Q gates visible to the algorithm,
        # makes much easier if don't have to deal with 1Q gates as successors
        self.requires = [Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]

    def _initialize_variables(self, dag):
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        self._max_iterations_without_progress = 10 * len(dag.qubits)  # Arbitrary.
        self._ops_since_progress = []
        self._extended_set = None
        self.dist_matrix = self.coupling_map.distance_matrix

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        self._mapped_dag = None
        if not self.fake_run:
            self._mapped_dag = dag.copy_empty_like()

        self._canonical_register = dag.qregs["q"]
        self._current_layout = Layout.generate_trivial_layout(self._canonical_register)
        self._bit_indices = {
            bit: idx for idx, bit in enumerate(self._canonical_register)
        }

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        self.qubits_decay = dict.fromkeys(dag.qubits, 1)

        # Start algorithm from the front layer and iterate until all gates done.
        # self._required_predecessors = super()._build_required_predecessors(dag)
        self._front_required_predecessors = self._build_required_predecessors(dag)
        self._intermediate_required_predecessors = self._build_required_predecessors(
            dag
        )
        self._num_search_steps = 0
        self._front_layer = dag.front_layer()
        self._intermediate_layer = []
        self._total_subs = 0

        self.rng = np.random.default_rng(self.seed)

    def _handle_no_progress(self):
        """Handle the case where no progress has been made in the last
        max_iterations_without_progress iterations."""
        # XXX, this methods are relying on updating variables via pass by reference
        # does this work for class variables?
        self._undo_operations(
            self._ops_since_progress, self._mapped_dag, self._current_layout
        )
        self._add_greedy_swaps(
            self._front_layer,
            self._mapped_dag,
            self._current_layout,
            self._canonical_register,
        )

    def _process_front_layer(self, dag):
        """Process the front layer of the DAG.

        Gates move from front layer into intermediate layer if a. has a
        physical topology edge b. predeccessors have been resolved (i.e.
        out of I layer), otherwise, gates remain in front layer.

        Gates move into the front layer if all their predecessors have
        left the front layer.
        """
        # Remove as many immediately applicable gates as possible
        # First, move gates from front layer to execute_gate_list
        execute_gate_list = []
        new_front_layer = []
        for node in self._front_layer:
            if len(node.qargs) == 2:
                v0, v1 = node.qargs
                if (
                    self.coupling_map.graph.has_edge(
                        self._current_layout._v2p[v0], self._current_layout._v2p[v1]
                    )
                    and self._intermediate_required_predecessors[node] == 0
                ):
                    execute_gate_list.append(node)
                else:
                    new_front_layer.append(node)
            else:
                execute_gate_list.append(node)
        self._front_layer = new_front_layer

        # check for if stuck

        # If no gates can be executed, add greedy swaps
        if (
            not execute_gate_list
            and len(self.ops_since_progress) > self._max_iterations_without_progress
        ):
            self._handle_no_progress(dag)
            return 1

        # Second, move execute_gate_list gates to intermediate layer
        # update front_required_predecessors, and refresh front layer
        if execute_gate_list:
            for node in execute_gate_list:
                self._intermediate_layer.append(node)

                # if can, move successors to front layer
                for successor in self._successors(node, dag):
                    self._front_required_predecessors[successor] -= 1
                    if self._front_required_predecessors[successor] == 0:
                        self._front_layer.append(successor)

                # XXX? should this be here or later?
                if node.qargs:
                    self._reset_qubits_decay()

                self.ops_since_progress = []
                self._extended_set = None

            # NOTE, I think this may be redundant, since forcing a stall
            # issue a return that triggers a continue in the main loop
            return 1

        return 0

    def _obtain_extended_set(self, dag, front_layer):
        # super() uses old convention
        self.required_predecessors = self._front_required_predecessors
        return super()._obtain_extended_set(dag, front_layer)

    def _process_intermediate_layer(self, dag):
        """Consider a virtual-swap substitution on every gate.

        If makes the topological distance cost from current layout to front layer better,
        then accept the change. NOTE, some changes might make the cost of decomposition worse.
        For example, CPhase with sqiswap basis changes from 2 to 3 gates. In this case, we decide
        to still accept since reducing the SWAP cost is more important.

        After considering a sub, move to mapped_dag and update intermediate_required_predecessors.
        NOTE: in rewrite we know only 2Q gates, and no dependencies inside of the intermediate layer.
        This means we can just eval the cost without needing to do any complicated bookkeeping.
        """

        extended_set = self._obtain_extended_set(dag, self._front_layer)

        trial_layout = self._current_layout.copy()
        for node in self._intermediate_layer:
            # use lookahead because these swaps are virtual - they have no cost related to parallelism
            no_sub_score = self._score_heuristic(
                "lookahead", self._front_layer, extended_set, trial_layout
            )

            # compare against the sub
            node_prime = _get_node_cns(node)
            trial_layout.swap(*node_prime.qargs)
            sub_score = self._score_heuristic(
                "lookahead", self._front_layer, extended_set, trial_layout
            )

            if sub_score < no_sub_score:
                self._total_subs += 1
                self._apply_gate(
                    self._mapped_dag, node_prime, trial_layout, self._canonical_register
                )
            else:
                # undo the changes to trial_layout
                trial_layout.swap(*node_prime.qargs)
                self._apply_gate(
                    self._mapped_dag, node, trial_layout, self._canonical_register
                )

            # update the intermediate required predecessors
            for successor in self._successors(node, dag):
                self._intermediate_required_predecessors[successor] -= 1

        self._intermediate_layer = []
        self._current_layout = trial_layout
        return 1

    def _find_and_insert_best_swap(self, dag):
        """Find the best swap and insert it into the DAG."""
        if self._extended_set is None:
            extended_set = self._obtain_extended_set(dag, self._front_layer)
        swap_scores = {}
        for swap_qubits in self._obtain_swaps(self._front_layer, self._current_layout):
            trial_layout = self._current_layout.copy()
            trial_layout.swap(*swap_qubits)
            score = self._score_heuristic(
                "decay", self._front_layer, extended_set, trial_layout, swap_qubits
            )
            swap_scores[swap_qubits] = score
        min_score = min(swap_scores.values())
        best_swaps = [k for k, v in swap_scores.items() if v == min_score]
        best_swaps.sort(
            key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]])
        )
        best_swap = self.rng.choice(best_swaps)
        swap_node = self._apply_gate(
            self._mapped_dag,
            DAGOpNode(op=SwapGate(), qargs=best_swap),
            self._current_layout,
            self._canonical_register,
        )
        self._current_layout.swap(*best_swap)
        self._ops_since_progress.append(swap_node)

        self._num_search_steps += 1
        if self._num_search_steps % DECAY_RESET_INTERVAL == 0:
            self._reset_qubits_decay()
        else:
            self.qubits_decay[best_swap[0]] += DECAY_RATE
            self.qubits_decay[best_swap[1]] += DECAY_RATE

    def run(self, dag):
        # Initialize variables
        self._initialize_variables(dag)

        # Main loop for processing front layer
        while self._front_layer:
            # process front layer
            # if true, repeat until no more gates can be processed
            if self._process_front_layer(dag):
                # XXX, this continue may not be necessary
                continue

            # If there are gates in the intermediate layer, process them
            if self._intermediate_layer:
                if self._process_intermediate_layer(dag):
                    continue

            # After all free gates are exhausted, find and insert the best swap
            self._find_and_insert_best_swap(dag)

        # Process remaining gates in the intermediate layer
        if self._intermediate_layer:
            for node in self._intermediate_layer:
                self._apply_gate(
                    self._mapped_dag,
                    node,
                    self._current_layout,
                    self._canonical_register,
                )
            self._intermediate_layer.remove(node)

        # assert front layer and intermediate layer are empty'
        assert not self._front_layer and not self._intermediate_layer

        # Set final layout and return the mapped dag
        self.property_set["final_layout"] = self._current_layout
        self.property_set["accepted_subs"] = self._total_subs
        return self._mapped_dag
