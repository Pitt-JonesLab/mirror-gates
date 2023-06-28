# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Routing via SWAP insertion using the SABRE method from Li et al."""

import logging
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import retworkx
from monodromy.depthPass import MonodromyDepth
from qiskit.circuit.library.standard_gates import SwapGate, iSwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import Unroller

# from concurrent.futures import ProcessPoolExecutor
from virtual_swap.cns_transform import _get_node_cns

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = (
    20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


# class TrialProcess(Process):
#     def __init__(self, parent, seed, dag, property_set, shared_dict):
#         super().__init__()
#         self.parent = parent
#         self.seed = seed
#         self.dag = dag
#         self.property_set = property_set
#         self.shared_dict = shared_dict

#     def run(self):
#         inner_rng = np.random.default_rng(int(self.seed * 2 ** 32))
#         trial_dag, trial_property_set = self.parent._nested_run(self.dag, self.property_set, inner_rng)
#         trial_cost = self.parent.calculate_gate_cost(trial_dag)
#         self.shared_dict[self.seed] = (trial_dag, trial_property_set, trial_cost)


class CNS_SabreSwap_V2(TransformationPass):
    r"""Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of the SWAP-based heuristic search from the SABRE qubit
    mapping paper [1] (Algorithm 1). The heuristic aims to minimize the number
    of lossy SWAPs inserted and the depth of the circuit.

    This algorithm starts from an initial layout of virtual qubits onto physical
    qubits, and iterates over the circuit DAG until all gates are exhausted,
    inserting SWAPs along the way. It only considers 2-qubit gates as only those
    are germane for the mapping problem (it is assumed that 3+ qubit gates are
    already decomposed).

    In each iteration, it will first check if there are any gates in the
    ``front_layer`` that can be directly applied. If so, it will apply them and
    remove them from ``front_layer``, and replenish that layer with new gates
    if possible. Otherwise, it will try to search for SWAPs, insert the SWAPs,
    and update the mapping.

    The search for SWAPs is restricted, in the sense that we only consider
    physical qubits in the neighborhood of those qubits involved in
    ``front_layer``. These give rise to a ``swap_candidate_list`` which is
    scored according to some heuristic cost function. The best SWAP is
    implemented and ``current_layout`` updated.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(
        self,
        coupling_map,
        heuristic="basic",
        seed=None,
        fake_run=False,
        preserve_layout=False,
        trials=1,
    ):
        r"""SabreSwap initializer.

        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic' or 'lookahead' or 'decay').
            seed (int): random seed used to tie-break among candidate swaps.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.

        Additional Information:

            The search space of possible SWAPs on physical qubits is explored
            by assigning a score to the layout that would result from each SWAP.
            The goodness of a layout is evaluated based on how viable it makes
            the remaining virtual gates that must be applied. A few heuristic
            cost functions are supported

            - 'basic':

            The sum of distances for corresponding physical qubits of
            interacting virtual qubits in the front_layer.

            .. math::

                H_{basic} = \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'lookahead':

            This is the sum of two costs: first is the same as the basic cost.
            Second is the basic cost but now evaluated for the
            extended set as well (i.e. :math:`|E|` number of upcoming successors to gates in
            front_layer F). This is weighted by some amount EXTENDED_SET_WEIGHT (W) to
            signify that upcoming gates are less important that the front_layer.

            .. math::

                H_{decay}=\frac{1}{\left|{F}\right|}\sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]
                    + W*\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'decay':

            This is the same as 'lookahead', but the whole cost is multiplied by a
            decay factor. This increases the cost if the SWAP that generated the
            trial layout was recently used (i.e. it penalizes increase in depth).

            .. math::

                H_{decay} = max(decay(SWAP.q_1), decay(SWAP.q_2)) {
                    \frac{1}{\left|{F}\right|} \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]\\
                    + W *\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]
                    }
        """

        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if coupling_map is None or coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()

        self.heuristic = heuristic
        self.preserve_layout = preserve_layout
        self.seed = seed
        self.fake_run = False
        self.required_predecessors = None
        self.qubits_decay = None
        self._bit_indices = None
        self.dist_matrix = None
        self.trials = trials

    # def run(self, dag):
    #     """Run with multiple trials in parallel."""
    #     best_dag, best_property_set = None, None
    #     best_cost = None
    #     rng = np.random.default_rng(self.seed)

    #     seeds = [int(rng.random() * 2 ** 32) for _ in range(self.trials)]

    #     manager = Manager()
    #     shared_dict = manager.dict()
    #     processes = [TrialProcess(self, seed, dag, self.property_set, shared_dict) for seed in seeds]

    #     for process in processes:
    #         process.start()

    #     for process in processes:
    #         process.join()

    #     for seed, (trial_dag, trial_property_set, trial_cost) in shared_dict.items():
    #         if best_cost is None or trial_cost < best_cost:
    #             best_dag, best_property_set = trial_dag, trial_property_set
    #             best_cost = trial_cost

    #     self.property_set = best_property_set
    #     self.property_set["best_cns_cost"] = best_cost
    #     return best_dag

    def run(self, dag):
        """Run with multiple trials.

        The original documentation, says to make this accept run that
        added the least number of swaps, but maybe we instead want to
        make this accept the run that has the lowest depth? This would
        be more expensive, because we would want depth to be calculated
        using our custom gate aware function.
        """
        best_dag, best_property_set = None, None
        best_cost = None
        rng = np.random.default_rng(self.seed)
        for _ in range(self.trials):
            inner_rng = np.random.default_rng(int(rng.random() * 2**32))
            trial_dag, trial_property_set = self._nested_run(
                dag, self.property_set, inner_rng
            )
            trial_cost = self.calculate_gate_cost(trial_dag)
            if best_cost is None or trial_cost < best_cost:
                best_dag, best_property_set = trial_dag, trial_property_set
                best_cost = trial_cost
        self.property_set = best_property_set
        self.property_set["best_cns_cost"] = best_cost
        # print("inner cost debug", best_cost)
        return best_dag

    # FIXME, this unroller needs to be removed, but because we haven't fully implemented using monodromy
    # the consolidate pass doesn't push together cns gates,
    # I actually don't know why, but unrolling before fixes this for the time being
    def calculate_gate_cost(self, dag: DAGCircuit):
        temp_dag = deepcopy(dag)
        unroller = Unroller(["cx", "u", "swap", "iswap"])
        temp_dag = unroller.run(temp_dag)
        depth_pass = MonodromyDepth(basis_gate=iSwapGate().power(1 / 2))
        depth_pass.property_set = unroller.property_set
        temp_dag = depth_pass.run(temp_dag)
        return depth_pass.property_set["monodromy_depth"]

    # FIXME, this could be way faster if using monodromy
    # rather than actually doing the decomposition, we just need to know the number of gates
    # def calculate_gate_cost(self, dag: DAGCircuit):
    #     """Force into sqiswap gates then calculate critical path cost."""
    #     temp_dag = deepcopy(dag)
    #     from qiskit.transpiler.passes import (
    #         Collect2qBlocks,
    #         ConsolidateBlocks,
    #         Unroller,
    #     )
    #     from slam.utils.transpiler_pass.weyl_decompose import RootiSwapWeylDecomposition

    #     unroller = Unroller(["cx", "u", "swap", "iswap"])
    #     # NOTE for some reason, iswap_primes don't get consolidated unless I unroll first
    #     collect = Collect2qBlocks()
    #     consolidate = ConsolidateBlocks(force_consolidate=True)
    #     weyl = RootiSwapWeylDecomposition()
    #     temp_dag = unroller.run(temp_dag)
    #     collect.property_set = unroller.property_set
    #     temp_dag = collect.run(temp_dag)
    #     consolidate.property_set = collect.property_set
    #     temp_dag = consolidate.run(temp_dag)
    #     weyl.property_set = consolidate.property_set
    #     temp_dag = weyl.run(temp_dag)
    #     for node in temp_dag.op_nodes():
    #         # only keep 2Q gates
    #         if len(node.qargs) != 2:
    #             temp_dag.remove_op_node(node)
    #     # print("debug calcualte_gate_cost",temp_dag.depth())
    #     return temp_dag.depth()

    def _nested_run(self, dag, trial_property_set, rng):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        max_iterations_without_progress = 10 * len(dag.qubits)  # Arbitrary.
        ops_since_progress = []
        extended_set = None

        # Normally this isn't necessary, but here we want to log some objects that have some
        # non-trivial cost to create.
        do_expensive_logging = logger.isEnabledFor(logging.DEBUG)

        self.dist_matrix = self.coupling_map.distance_matrix

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None
        # XXX, tracking snaps to be appending to property_set at the end

        if not self.fake_run:
            mapped_dag = dag.copy_empty_like()

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
        # I thought this needed to start initial layout with whatever has applied in ApplyLayout
        # NOTE, I thought I needed to do this, but actually the circuit already has been transformed
        # so start with trivial layout, because gates have been rearranged for physical indexing
        # current_layout = self.property_set["layout"].copy()
        # self._bit_indices = current_layout.get_virtual_bits()

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        self.qubits_decay = dict.fromkeys(dag.qubits, 1)

        # Start algorithm from the front layer and iterate until all gates done.
        self.required_predecessors = self._build_required_predecessors(dag)
        num_search_steps = 0
        front_layer = dag.front_layer()
        intermediate_layer = []
        total_subs = 0

        while front_layer:
            # XXX
            # printing mapped_dag at each iteration
            # if mapped_dag is not None:
            #     mapped_dag_snaps.append(mapped_dag.draw())

            execute_gate_list = []

            # Remove as many immediately applicable gates as possible
            new_front_layer = []

            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    # Accessing layout._v2p directly to avoid overhead from __getitem__ and a
                    # single access isn't feasible because the layout is updated on each iteration
                    if self.coupling_map.graph.has_edge(
                        current_layout._v2p[v0], current_layout._v2p[v1]
                    ):
                        execute_gate_list.append(node)
                    else:
                        new_front_layer.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)

            front_layer = new_front_layer

            if (
                not execute_gate_list
                and len(ops_since_progress) > max_iterations_without_progress
            ):
                # print("HERE, backtracking")
                # Backtrack to the last time we made progress, then greedily insert swaps to route
                # the gate with the smallest distance between its arguments.  This is a release
                # valve for the algorithm to avoid infinite loops only, and should generally not
                # come into play for most circuits.
                self._undo_operations(ops_since_progress, mapped_dag, current_layout)
                self._add_greedy_swaps(
                    front_layer, mapped_dag, current_layout, canonical_register
                )
                continue

            if execute_gate_list:
                for node in execute_gate_list:
                    # instead of adding to mapped_dag, add to intermediate layer
                    intermediate_layer.append(node)
                    # add its successors to front layer
                    for successor in self._successors(node, dag):
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)
                    if node.qargs:
                        self._reset_qubits_decay()
                ops_since_progress = []
                extended_set = None
                continue

            # HERE, we have completed building the front layer
            # move things from interemediate layer into mapped dag

            # clean the intermediate layer, fixes 1Q gates that follow possible cns subs
            temp_1q_layer = []
            if intermediate_layer:
                for node in intermediate_layer:
                    # FIXME, bad code repetition
                    if len(node.qargs) != 2:
                        continue
                    node_successors = list(
                        filter(
                            lambda x: isinstance(x, DAGOpNode) and len(x.qargs) == 2,
                            list(dag.descendants(node)),
                        )
                    )
                    # check that none of the node's successors are in intermediate_layer
                    if any(
                        [
                            successor in intermediate_layer
                            for successor in node_successors
                        ]
                    ):
                        continue
                    # now, we know this 2Q is a CNS sub candidate,
                    # so need to move its 1Q gates if in intermediate_layer
                    # move them to front_layer
                    node_successors = list(
                        filter(
                            lambda x: isinstance(x, DAGOpNode) and len(x.qargs) == 1,
                            list(dag.descendants(node)),
                        )
                    )
                    for successor in node_successors:
                        if successor in intermediate_layer:
                            temp_1q_layer.append(successor)
                            intermediate_layer.remove(successor)

            if intermediate_layer:
                # print(intermediate_layer)
                temp_intermediate_layer = intermediate_layer.copy()
                trial_layout = current_layout.copy()
                for node in temp_intermediate_layer:
                    # if is a leaf node (in intermediate_layer), eval the cns subs
                    # remove 1Q gates from node_successors list
                    node_successors = list(
                        filter(
                            lambda x: isinstance(x, DAGOpNode) and len(x.qargs) == 2,
                            list(dag.descendants(node)),
                        )
                    )
                    # check that none of the node's successors are in intermediate_layer
                    if any(
                        [
                            successor in intermediate_layer
                            for successor in node_successors
                        ]
                    ):
                        self._apply_gate(
                            mapped_dag, node, current_layout, canonical_register
                        )

                    elif node.name not in ["iswap", "cx"]:
                        self._apply_gate(
                            mapped_dag, node, current_layout, canonical_register
                        )
                    else:
                        extended_set = self._obtain_extended_set(dag, front_layer)
                        # check score on trial_layout
                        no_sub_score = self._score_heuristic(
                            "lookahead", front_layer, extended_set, trial_layout
                        )
                        # compare against score on node_prime
                        node_prime = _get_node_cns(node)
                        trial_layout.swap(*node_prime.qargs)
                        sub_score = self._score_heuristic(
                            "lookahead", front_layer, extended_set, trial_layout
                        )

                        temp_accept_flag = False
                        if sub_score == no_sub_score:
                            # 50% chance of accept
                            if rng.random() < 0.5:
                                temp_accept_flag = True
                        if sub_score < no_sub_score or temp_accept_flag:
                            # apply the sub
                            self._apply_gate(
                                mapped_dag,
                                node_prime,
                                current_layout,
                                canonical_register,
                            )
                            total_subs += 1
                        else:
                            # undo changes to trial_layout, apply the original gate
                            trial_layout.swap(*node_prime.qargs)
                            self._apply_gate(
                                mapped_dag, node, current_layout, canonical_register
                            )

                    # clear intermediate_layer
                    intermediate_layer.remove(node)
                # current_layout <- trial_layout
                current_layout = trial_layout

                # put on 1Q gates from temp_1q_layer
                for node in temp_1q_layer:
                    self._apply_gate(
                        mapped_dag, node, current_layout, canonical_register
                    )
                extended_set = None
                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            if extended_set is None:
                extended_set = self._obtain_extended_set(dag, front_layer)
            swap_scores = {}
            for swap_qubits in self._obtain_swaps(front_layer, current_layout):
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)
                score = self._score_heuristic(
                    "decay", front_layer, extended_set, trial_layout, swap_qubits
                )
                swap_scores[swap_qubits] = score
            min_score = min(swap_scores.values())
            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(
                key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]])
            )
            best_swap = rng.choice(best_swaps)
            swap_node = self._apply_gate(
                mapped_dag,
                DAGOpNode(op=SwapGate(), qargs=best_swap),
                current_layout,
                canonical_register,
            )
            current_layout.swap(*best_swap)
            ops_since_progress.append(swap_node)

            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL == 0:
                self._reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE

            # Diagnostics
            if do_expensive_logging:
                logger.debug("SWAP Selection...")
                logger.debug(
                    "extended_set: %s", [(n.name, n.qargs) for n in extended_set]
                )
                logger.debug("swap scores: %s", swap_scores)
                logger.debug("best swap: %s", best_swap)
                logger.debug("qubits decay: %s", self.qubits_decay)

        # empty intermediate layer one last time
        if intermediate_layer:
            for node in intermediate_layer:
                self._apply_gate(mapped_dag, node, current_layout, canonical_register)
            intermediate_layer.remove(node)

        trial_property_set["final_layout"] = current_layout

        # XXX
        # self.property_set["mapped_dag_snaps"] = mapped_dag_snaps
        trial_property_set["accept_subs"] = total_subs

        if not self.fake_run:
            return mapped_dag, trial_property_set
        return dag, trial_property_set

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(
            new_node.op, new_node.qargs, new_node.cargs
        )

    def _reset_qubits_decay(self):
        """Reset all qubit decay factors to 1 upon request (to forget about
        past penalizations)."""
        self.qubits_decay = {k: 1 for k in self.qubits_decay.keys()}

    def _build_required_predecessors(self, dag):
        out = defaultdict(int)
        # We don't need to count in- or out-wires: outs can never be predecessors, and all input
        # wires are automatically satisfied at the start.
        for node in dag.op_nodes():
            for successor in self._successors(node, dag):
                out[successor] += 1
        return out

    def _successors(self, node, dag):
        """Return an iterable of the successors along each wire from the given
        node.

        This yields the same successor multiple times if there are
        parallel wires (e.g. two adjacent operations that have one clbit
        and qubit in common), which is important in the swapping
        algorithm for detecting if each wire has been accounted for.
        """
        for _, successor, _ in dag.edges(node):
            if isinstance(successor, DAGOpNode):
                yield successor

    def _is_resolved(self, node):
        """Return True if all of a node's predecessors in dag are applied."""
        return self.required_predecessors[node] == 0

    def _obtain_extended_set(self, dag, front_layer):
        """Populate extended_set by looking ahead a fixed number of gates.

        For each existing element add a successor until reaching limit.
        """
        extended_set = []
        decremented = []
        tmp_front_layer = front_layer
        done = False
        while tmp_front_layer and not done:
            new_tmp_front_layer = []
            for node in tmp_front_layer:
                for successor in self._successors(node, dag):
                    decremented.append(successor)
                    self.required_predecessors[successor] -= 1
                    if self._is_resolved(successor):
                        new_tmp_front_layer.append(successor)
                        if len(successor.qargs) == 2:
                            extended_set.append(successor)
                if len(extended_set) >= EXTENDED_SET_SIZE:
                    done = True
                    break
            tmp_front_layer = new_tmp_front_layer
        for node in decremented:
            self.required_predecessors[node] += 1
        return extended_set

    def _obtain_swaps(self, front_layer, current_layout):
        """Return a set of candidate swaps that affect qubits in front_layer.

        For each virtual qubit in front_layer, find its current location
        on hardware and the physical qubits in that neighborhood. Every
        SWAP on virtual qubits that corresponds to one of those physical
        couplings is a candidate SWAP.

        Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not
        duplicated.
        """
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted(
                        [virtual, virtual_neighbor], key=lambda q: self._bit_indices[q]
                    )
                    candidate_swaps.add(tuple(swap))
        return candidate_swaps

    def _add_greedy_swaps(self, front_layer, dag, layout, qubits):
        """Mutate ``dag`` and ``layout`` by applying greedy swaps to ensure that at least one gate
        can be routed."""
        layout_map = layout._v2p
        target_node = min(
            front_layer,
            key=lambda node: self.dist_matrix[
                layout_map[node.qargs[0]], layout_map[node.qargs[1]]
            ],
        )
        for pair in _shortest_swap_path(
            tuple(target_node.qargs), self.coupling_map, layout
        ):
            self._apply_gate(dag, DAGOpNode(op=SwapGate(), qargs=pair), layout, qubits)
            layout.swap(*pair)

    def _compute_cost(self, layer, layout):
        cost = 0
        layout_map = layout._v2p
        for node in layer:
            cost += self.dist_matrix[
                layout_map[node.qargs[0]], layout_map[node.qargs[1]]
            ]
        return cost

    def _score_heuristic(
        self, heuristic, front_layer, extended_set, layout, swap_qubits=None
    ):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign
        a cost to it. The goodness of a layout is evaluated based on how
        viable it makes the remaining virtual gates that must be
        applied.
        """
        first_cost = self._compute_cost(front_layer, layout)
        if heuristic == "basic":
            return first_cost

        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return (
                max(
                    self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]]
                )
                * total_cost
            )

        raise TranspilerError("Heuristic %s not recognized." % heuristic)

    def _undo_operations(self, operations, dag, layout):
        """Mutate ``dag`` and ``layout`` by undoing the swap gates listed in ``operations``."""
        if dag is None:
            for operation in reversed(operations):
                layout.swap(*operation.qargs)
        else:
            for operation in reversed(operations):
                dag.remove_op_node(operation)
                p0 = self._bit_indices[operation.qargs[0]]
                p1 = self._bit_indices[operation.qargs[1]]
                layout.swap(p0, p1)


def _transform_gate_for_layout(op_node, layout, device_qreg):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = copy(op_node)
    mapped_op_node.qargs = tuple(device_qreg[layout._v2p[x]] for x in op_node.qargs)
    return mapped_op_node


def _shortest_swap_path(target_qubits, coupling_map, layout):
    """Return an iterator that yields the swaps between virtual qubits needed to bring the two
    virtual qubits in ``target_qubits`` together in the coupling map."""
    v_start, v_goal = target_qubits
    start, goal = layout._v2p[v_start], layout._v2p[v_goal]
    # TODO: remove the list call once using retworkx 0.12, as the return value can be sliced.
    path = list(
        retworkx.dijkstra_shortest_paths(coupling_map.graph, start, target=goal)[goal]
    )
    # Swap both qubits towards the "centre" (as opposed to applying the same swaps to one) to
    # parallelise and reduce depth.
    split = len(path) // 2
    forwards, backwards = path[1:split], reversed(path[split:-1])
    for swap in forwards:
        yield v_start, layout._p2v[swap]
    for swap in backwards:
        yield v_goal, layout._p2v[swap]
