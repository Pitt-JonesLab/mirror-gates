"""V3 Rewrite of CNS SABRE Implementation."""

import copy
import multiprocessing as mp

import numpy as np
from monodromy.depthPass import MonodromyDepth
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import SwapGate
from qiskit.circuit.library.standard_gates import iSwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TranspilerError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout

from mirror_gates.cns_transform import _get_node_cns

# NOTE, the current qiskit version uses a rust backend
# we have in deprecated/ the original python implementation we modified in V2.
# might be useful to grab some functions from it
from mirror_gates.qiskit.sabre_swap import SabreSwap as LegacySabreSwap
from mirror_gates.utilities import DoNothing

EXTENDED_SET_SIZE = (
    20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.
DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


class ParallelSabreSwapMS(TransformationPass):
    """Parallel version of SabreSwapMS."""

    def __init__(
        self,
        coupling_map,
        heuristic="lookahead",
        trials=20,
        basis_gate=None,
        parallel=True,
        seed=None,
        use_fast_settings=True,
        cost_function="depth",
        fixed_aggression=None,
    ):
        """Initialize the pass."""
        super().__init__()
        self.coupling_map = coupling_map
        self.heuristic = heuristic
        self.parallel = parallel
        self.fake_run = False
        self.use_fast_settings = use_fast_settings
        self.anneal_index = 1.0
        self.fixed_aggression = fixed_aggression

        self.num_trials = trials
        if self.num_trials < 4:
            raise TranspilerError("Use at least 4 trials for SabreSwapMS.")

        # NOTE, normally is required but we make sure is in the pre-stage
        # this is so we don't have to call this function in each layout_trial
        # self.requires = [FastConsolidateBlocks(coord_caching=True)]

        self.cost_function = cost_function
        self.basis_gate = basis_gate or iSwapGate().power(1 / 2)
        self.cost_pass = MonodromyDepth(
            consolidate=False,
            basis_gate=self.basis_gate,
            use_fast_settings=self.use_fast_settings,
        )
        # we only use this to generate seeds
        rng = np.random.default_rng(seed)
        # generate an array of seeds
        self.seeds = [rng.integers(0, 2**32 - 1) for _ in range(self.num_trials)]

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on `dag`."""
        if self.parallel:
            mapped_dag = self._parallel_run(dag)
        else:
            mapped_dag = self._serial_run(dag)
        return mapped_dag if not self.fake_run else dag

    def set_anneal_params(self, fb_iter: float):
        """Anneal configuration to escape local minima.

        Args: fb_iter (float): fraction of current iteration to total iterations

        In the layout pass, each forward/backwards mapping calls routing multiple times,
        and keeps the one that minimized the cost function. Instead of keeping the min,
        select randomly from the top % performers. Then for each successive iteration,
        select from smaller % of the top scores until only keep the best score circuit.
        """
        self.anneal_index = fb_iter

    def _anneal_select_result(self, results):
        """Select a result stochastically based on annealing parameters.

        Args:
            results (list): each a tuple with cost as the first element.

        Returns:
            tuple: selected result.
        """
        # Determine how many of the top results to consider, based on annealing.
        # Early on, consider more results. Later, consider fewer.
        n_top_results = max(int((0.5 - 0.5 * self.anneal_index) * len(results)), 1)

        # Sort results by cost and select the top ones.
        top_results = sorted(results, key=lambda x: x[0])[:n_top_results]

        # Select one of the top results randomly.
        random_index = np.random.randint(0, len(top_results))
        return top_results[random_index]

    def _parallel_run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass in parallel."""
        self.dag = dag  # Store the dag can be accessed by run_single_trial
        with mp.Pool() as pool:
            results = pool.map(self._run_single_trial, range(self.num_trials))
        # best_score, best_result, best_property_set = min(results, key=lambda x: x[0])
        best_score, best_result, best_property_set = self._anneal_select_result(results)
        self.property_set["final_layout"] = best_property_set["final_layout"]
        self.property_set["accepted_subs"] = best_property_set["accepted_subs"]
        self.property_set["best_score"] = best_score
        return best_result

    def _serial_run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass in serial."""
        self.dag = dag  # Store the dag can be accessed by run_single_trial
        results = [self._run_single_trial(i) for i in range(self.num_trials)]
        # best_score, best_result, best_property_set = min(results, key=lambda x: x[0])
        best_score, best_result, best_property_set = self._anneal_select_result(results)
        self.property_set["final_layout"] = best_property_set["final_layout"]
        self.property_set["accepted_subs"] = best_property_set["accepted_subs"]
        self.property_set["best_score"] = best_score
        return best_result

    def _run_single_trial(self, trial_number):
        """Run a single trial of the pass."""
        if self.fixed_aggression is not None:
            aggression = self.fixed_aggression
        else:
            aggression = 3  # Default aggression level
            if trial_number < 0.15 * self.num_trials:
                aggression = 0
            elif trial_number < 0.50 * self.num_trials:  # 0.15 + 0.35
                aggression = 1
            elif trial_number < 0.85 * self.num_trials:  # 0.50 + 0.35
                aggression = 2
        trial = SabreSwapMS(
            self.coupling_map,
            self.heuristic,
            self.property_set,
            aggression=aggression,
            use_fast_settings=self.use_fast_settings,
        )
        trial.seed = self.seeds[trial_number]
        result = trial.run(self.dag)
        score = self._calculate_score(result, trial.property_set)
        return score, result, trial.property_set

    def _calculate_score(self, result: DAGCircuit, property_set) -> float:
        """Calculate the score of a result."""
        # assuming consolidation is done before SabreSwapMS
        # NOTE, means CNS subs need to be single gate
        # unroller = Unroller(["cx", "iswap", "u", "swap"])
        if self.cost_function == "depth":
            self.cost_pass.property_set = property_set
            self.cost_pass.run(result)
            # print(f"monodromy_depth: \
            #       {self.cost_pass.property_set['monodromy_depth']}")
            return self.cost_pass.property_set["monodromy_depth"]
        else:  # basic SABRE
            do_nothing = DoNothing()
            do_nothing.property_set = property_set
            do_nothing.run(result)
            return do_nothing.property_set["required_swaps"]


class SabreSwapMS(LegacySabreSwap):
    """V3 Rewrite of CNS SABRE Implementation."""

    def __init__(
        self,
        coupling_map,
        property_set,
        heuristic="lookahead",
        aggression=2,
        use_fast_settings=True,
    ):
        """Initialize the pass.

        Args:
            aggression (int): How aggressively to search for virtual swaps.
                0: No virtual-swaps are accepted.
                1: Only virtual-swaps that improve the cost.
                2: Virtual-swaps that improve or do not change the cost.
                3: All virtual-swaps are accepted.
        """
        # deepcopy for safety
        if aggression not in [0, 1, 2, 3]:
            raise ValueError("Invalid aggression level.")
        self.aggression = aggression
        self.property_set = copy.deepcopy(property_set)
        super().__init__(coupling_map, heuristic=heuristic)
        self.use_fast_settings = use_fast_settings

    def _initialize_variables(self, dag):
        """Initialize variables for the algorithm."""
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
        self._considered_subs = 0
        self._required_swaps = 0

        self.rng = np.random.default_rng(self.seed)

    def _handle_no_progress(self, dag):
        """Handle the case where no progress is being made."""
        # XXX, this methods are relying on updating variables via pass by reference
        # does this work for class variables?
        raise NotImplementedError("Unresovled bug in handle_no_progress function")
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

        Gates move from front layer into intermediate layer if a. has a physical
        topology edge b. predeccessors have been resolved (i.e. out of I layer),
        otherwise, gates remain in front layer.

        Gates move into the front layer if all their predecessors have left the front
        layer.
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
            and len(self._ops_since_progress) > self._max_iterations_without_progress
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

            self._ops_since_progress = []
            self._extended_set = None

            # NOTE, I think this may be redundant, since forcing a stall
            # issue a return that triggers a continue in the main loop
            return 1

        return 0

    def _obtain_extended_set(self, dag, front_layer):
        """Obtain the extended set of gates for the lookahead window."""
        # super() uses old convention
        self.required_predecessors = self._front_required_predecessors
        return super()._obtain_extended_set(dag, front_layer)

    def _process_intermediate_layer(self, dag):
        """Consider a mirror-gate substitution on every gate.

        If makes the topological distance cost to front layer better, then accept the
        change. NOTE, some changes might make decomposition cost worse. For example,
        CPhase with sqiswap basis changes from 2 to 3 gates. Then, decide to still
        accept since reducing the SWAP cost is more important.

        After, move to mapped_dag and update intermediate_required_predecessors. in
        rewrite we know only 2Q gates, and no dependencies in the intermediate layer.
        means we can just eval the cost without needing to do any messy bookkeeping.
        """
        if self._extended_set is None:
            extended_set = self._obtain_extended_set(dag, self._front_layer)

        # XXX instead of all, just pop the first one
        # for node in [self._intermediate_layer.pop(0)]:
        for node in self._intermediate_layer:
            # handles barriers, measure, reset, etc.
            if not isinstance(node.op, Gate):
                self._apply_gate(
                    self._mapped_dag,
                    node,
                    self._current_layout,
                    self._canonical_register,
                )
                continue

            # use lookahead (instead of decay) because these swaps are virtual
            strategy = "lookahead"
            # they have no cost related to parallelism
            no_sub_score = self._score_heuristic(
                strategy, self._front_layer, extended_set, self._current_layout
            )

            # compare against the sub
            node_prime = _get_node_cns(node, self.use_fast_settings)
            self._current_layout.swap(*node.qargs)

            sub_score = self._score_heuristic(
                strategy, self._front_layer, extended_set, self._current_layout
            )
            self._considered_subs += 1

            if (
                (self.aggression == 1 and sub_score < no_sub_score)
                or (self.aggression == 2 and sub_score <= no_sub_score)
                or (self.aggression == 3)
            ):
                self._total_subs += 1
                self._apply_gate(
                    self._mapped_dag,
                    node_prime,
                    self._current_layout,
                    self._canonical_register,
                )

            else:
                # undo the changes to trial_layout
                self._current_layout.swap(*node.qargs)
                self._apply_gate(
                    self._mapped_dag,
                    node,
                    self._current_layout,
                    self._canonical_register,
                )

            # update the intermediate required predecessors
            for successor in self._successors(node, dag):
                self._intermediate_required_predecessors[successor] -= 1

        self._intermediate_layer = []
        return 1

    # TODO, function that makes it less likely to accept a SWAP on a contested edge
    # e.g. in tree-hierachy, want to have less SWAPs on the high-tier "bottleneck" edges

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
        # NOTE- hardcode the SWAP monodromy coordinates
        # avoids needing to call unitary_to_coordinate function later
        if self.use_fast_settings:  # questionable
            swap_node.op._monodromy_coord = [0.25, 0.25, 0.25, -0.75]
        self._current_layout.swap(*best_swap)
        self._ops_since_progress.append(swap_node)

        self._num_search_steps += 1
        if self._num_search_steps % DECAY_RESET_INTERVAL == 0:
            self._reset_qubits_decay()
        else:
            self.qubits_decay[best_swap[0]] += DECAY_RATE
            self.qubits_decay[best_swap[1]] += DECAY_RATE
        self._required_swaps += 1

    def run(self, dag):
        """Run the pass on `dag`."""
        # Initialize variables
        self._initialize_variables(dag)

        # Main loop for processing front layer
        while self._front_layer:
            # process front layer
            # if true, repeat until no more gates can be processed
            if self._process_front_layer(dag):
                # XXX, this continue may not be necessary
                # keep for safety :)
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
            self._intermediate_layer = []
        # assert front layer and intermediate layer are empty'
        assert not self._front_layer and not self._intermediate_layer

        # Set final layout and return the mapped dag
        self.property_set["final_layout"] = self._current_layout

        # write accepted_subs as fraction of total number of 2Q gates considered
        if self._considered_subs == 0:
            self.property_set["accepted_subs"] = 0
        else:
            self.property_set["accepted_subs"] = (
                1.0 * self._total_subs / self._considered_subs
            )
        self.property_set["required_swaps"] = self._required_swaps

        return self._mapped_dag
