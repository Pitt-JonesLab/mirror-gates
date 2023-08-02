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


class ParallelMirage(TransformationPass):
    """Parallel version of Mirage."""

    def __init__(
        self,
        coupling_map,
        heuristic="lookahead",
        trials=6,
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
        self.anneal_index = None
        self.fixed_aggression = fixed_aggression
        self.atomic_routing = Mirage

        self.num_trials = trials
        if self.num_trials < 4:
            raise TranspilerError("Use at least 4 trials for Mirage.")

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
        # set dag as attribute for use in _run_single_trial
        self.dag = dag

        # run the pass over trials
        if self.parallel:
            with mp.Pool() as pool:
                results = pool.map(self._run_single_trial, range(self.num_trials))
        else:
            results = map(self._run_single_trial, range(self.num_trials))

        # find the best result
        best_score, best_result, best_property_set = min(results, key=lambda x: x[0])

        # handle property_set to get best attributes
        self.property_set["final_layout"] = best_property_set["final_layout"]
        self.property_set["accepted_subs"] = best_property_set["accepted_subs"]
        self.property_set["best_score"] = best_score

        return best_result if not self.fake_run else dag

    def set_anneal_params(self, fb_iter: float):
        """Anneal configuration to escape local minima.

        Args: fb_iter (float): fraction of current iteration to total iterations
            0 means high temperature, 1 means low temperature

        In the layout pass, each forward/backwards mapping calls routing multiple times.
        Use this index to adjust annealing parameters.
        """
        self.anneal_index = fb_iter

    def _run_single_trial(self, trial_number):
        """Run a single trial of the pass.

        There are 2 contrasting modes of operation. First, using aggression levels which
        are fixed for the entire trial. These define the threshold for accepting a
        virtual-swap. Second, is annealing, where instead of using a fixed threshold, we
        define a probability function for accepting any virtual-swap.

        Args:
            trial_number (int): The trial number.

        Returns:
            float: The score of the result.
            DAGCircuit: The mapped DAG.
            dict: The property set of the trial.
        """
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
        trial = self.atomic_routing(
            self.coupling_map,
            self.heuristic,
            property_set=self.property_set,
            aggression=aggression,
            anneal_index=self.anneal_index,
            use_fast_settings=self.use_fast_settings,
        )
        trial.seed = self.seeds[trial_number]
        result = trial.run(self.dag)
        score = self._calculate_score(result, trial.property_set)
        return score, result, trial.property_set

    def _calculate_score(self, result: DAGCircuit, property_set) -> float:
        """Calculate the score of a result.

        Assumes that circuit preserves consolidation. Meaning, the circuit is
        composed of no sequential 2Q gates. Following CNS substitution rules, this
        means rather than inserting U+SWAP gates, we insert the mirror U' gates.

        Args:
            result (DAGCircuit): The result of the trial.
            property_set (dict): The property set of the trial.

        Returns:
            float: The score of the result.
        """
        if self.cost_function == "depth":
            self.cost_pass.property_set = property_set
            self.cost_pass.run(result)
            # print(f"monodromy_depth: \
            #       {self.cost_pass.property_set['monodromy_depth']}")
            return self.cost_pass.property_set["monodromy_depth"]
        else:  # basic SABRE
            # use DoNothing pass to extract trial property_set from result
            # FIXME is this necessary?
            do_nothing = DoNothing()
            do_nothing.property_set = property_set
            do_nothing.run(result)
            return do_nothing.property_set["required_swaps"]


class Mirage(LegacySabreSwap):
    """V3 Rewrite of CNS SABRE Implementation."""

    @staticmethod
    def probabilistic_acceptance(temperature, cost0, cost1):
        """Return the probability of accepting a virtual-swap.

        Args:
            temperature (float): Temperature of the system.
            cost0 (float): Cost of the current layout.
            cost1 (float): Cost of the trial layout.
        """
        if cost1 < cost0:
            return 1.0
        else:
            p = np.exp((cost0 - cost1) / temperature)
            # print(f"Cost0: {cost0}, Cost1: {cost1}, p: {p}")
            return p

    def __init__(
        self,
        coupling_map,
        heuristic="lookahead",
        property_set=None,
        aggression=2,
        use_fast_settings=True,
        anneal_index=None,
    ):
        """Initialize the pass.

        Args:
            aggression (int): How aggressively to search for virtual swaps.
                0: No virtual-swaps are accepted.
                1: Only virtual-swaps that improve the cost.
                2: Virtual-swaps that improve or do not change the cost.
                3: All virtual-swaps are accepted.
            anneal_index (float): defined as (0,1] progress through total iterations.
                If anneal_index is None, then use the fixed aggression level.
                Otherwise, use the annealing function to determine the probability
                of accepting a virtual-swap.
        """
        # deepcopy for safety
        if aggression not in [0, 1, 2, 3]:
            raise ValueError("Invalid aggression level.")
        self.aggression = aggression
        self.anneal_index = anneal_index
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
        assert self.fake_run is False  # real run required to evaluate cost function
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

    # TODO:
    # tie-breaks when routing costs are equivalent could be broken using decomp cost
    # if routing is equiv, then follows lower depth comes from cheaper unitary blocks
    # however, for sqiswap gates this would only make a difference on QV circuits
    # this is because if input circuits are entirely CPhase gates,
    # then we know the original gate will be cheaper than mirror (for sqiswap basis)

    def _accept_virtual_swap(self, sub_score, no_sub_score):
        """Decide whether to accept a virtual swap based on aggression or annealing.

        Args:
            sub_score (float): Score after the virtual swap.
            no_sub_score (float): Score without the virtual swap.

        Returns:
            bool: Whether to accept the virtual swap.
        """
        # If annealing is enabled
        if self.anneal_index is not None:
            # Calculate the "temperature" for simulated annealing
            temperature = max(0.01, 1.0 - self.anneal_index)
            # print("temperature", temperature)

            # Use the probabilistic accept function to decide whether to accept the swap
            return (
                self.probabilistic_acceptance(temperature, no_sub_score, sub_score)
                > self.rng.random()
            )
        else:
            # If annealing is not enabled, use the aggression level logic
            return (
                (self.aggression == 1 and sub_score < no_sub_score)
                or (self.aggression == 2 and sub_score <= no_sub_score)
                or (self.aggression == 3)
            )

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

            if self._accept_virtual_swap(sub_score, no_sub_score):
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

        # WARNING: not well understood behavior
        # fix leading SWAPs
        while any([op.name == "swap" for op in self._mapped_dag.front_layer()]):
            for op in self._mapped_dag.front_layer():
                if op.name == "swap":
                    self._mapped_dag.remove_op_node(op)

        # check if mapped_dag front layer contains swaps
        if not self.fake_run and any(
            [op.name == "swap" for op in self._mapped_dag.front_layer()]
        ):
            raise TranspilerError("Mirage begins with a SWAP.")

        # write accepted_subs as fraction of total number of 2Q gates considered
        if self._considered_subs == 0:
            self.property_set["accepted_subs"] = 0
        else:
            self.property_set["accepted_subs"] = (
                1.0 * self._total_subs / self._considered_subs
            )
        self.property_set["required_swaps"] = self._required_swaps

        return self._mapped_dag
