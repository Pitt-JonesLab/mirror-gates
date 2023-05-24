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

# XXX
from itertools import permutations

import numpy as np
import rustworkx as retworkx
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout

from virtual_swap.cns_transform import _get_node_cns

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = (
    20  # Size of lookahead window. TODO: set dynamically to len(current_layout)
)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


class CNS_SabreSwap(TransformationPass):
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
        # XXX
        # # CNS assumes all inputs are either CX or iSWAP gates
        # self.basis = ["u", "cx", "iswap"]
        # self.requires.append(Unroller(self.basis))

        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if coupling_map is None or coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()

        self.heuristic = heuristic
        self.seed = seed
        self.fake_run = fake_run
        self.required_predecessors = None
        self.qubits_decay = None
        self._bit_indices = None
        self.dist_matrix = None

    def run(self, dag):
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

        rng = np.random.default_rng(self.seed)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None

        # XXX, tracking snaps to be appending to property_set at the end
        mapped_dag_snaps = []

        if not self.fake_run:
            mapped_dag = dag.copy_empty_like()

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)

        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        self.qubits_decay = dict.fromkeys(dag.qubits, 1)

        # Start algorithm from the front layer and iterate until all gates done.
        self.required_predecessors = self._build_required_predecessors(dag)
        num_search_steps = 0
        front_layer = dag.front_layer()

        # XXX
        # tracking CNS sub candidates
        # these lists are outside of while, so don't get reset on `continue`
        # clear them manually once things get pushed into mapped_dag
        total_subs = 0
        outer_cns_subs = []
        outer_execute_gate_list = []

        while front_layer:
            # XXX
            # printing mapped_dag at each iteration
            if mapped_dag is not None:
                mapped_dag_snaps.append(mapped_dag.draw())

            execute_gate_list = []
            cns_sub_candidates = []

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
                        # execute_gate_list.append(node)
                        # XXX
                        # Rather than appending a gate immediately to execute_gate_list
                        # we want to track nodes that could potentially have a CNS
                        if node.name in ["cx", "iswap"]:
                            # track nodes in separate list
                            cns_sub_candidates.append(node)
                        else:
                            execute_gate_list.append(node)
                        # XXX

                    else:
                        new_front_layer.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)
            front_layer = new_front_layer

            # rearranged, so that front_layer is formed before eval
            # execute_gate_list node's successors are pushed into front_layer
            # NOTE, we don't know if node in execute_gate_list will be node or node_prime
            # but the successors of node don't change
            find_successors = execute_gate_list + cns_sub_candidates
            if find_successors:
                for node in find_successors:
                    for successor in self._successors(node, dag):
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)

                    if node.qargs:
                        self._reset_qubits_decay()
                #
                outer_cns_subs.extend(cns_sub_candidates)
                outer_execute_gate_list.extend(execute_gate_list)
                continue

            # XXX
            # new CNS trial_layout evaluation block,
            # given a fully formed front_layer node list
            # now we want to iteratively find if layout swaps from CNS are better
            # NOTE could do 2^n permutations for n CNS subs,
            # instead, we do iteratively and accept a layout swap if improves score

            # no more continues
            cns_sub_candidates = outer_cns_subs
            outer_cns_subs = []

            for node in front_layer:
                if len(node.qargs) != 2:
                    raise ValueError("bad front layer")

            if front_layer:
                # initialize
                if extended_set is None:
                    extended_set = self._obtain_extended_set(dag, front_layer)

                for node in extended_set:
                    if len(node.qargs) != 2:
                        raise ValueError("bad extended")

                accept_subs = {node: False for node in cns_sub_candidates}
                best_score = self._score_heuristic(
                    "lookahead", front_layer, extended_set, current_layout
                )
                trial_layout = current_layout.copy()

                # iteratively test CNS subs
                # for node in cns_sub_candidates:
                #     trial_layout.swap(*node.qargs)
                #     score = self._score_heuristic(
                #         "lookahead", front_layer, extended_set, trial_layout
                #     )
                #     if score < best_score:
                #         best_score = score
                #         accept_subs[node] = True
                #     else:
                #         # undo the layout swap
                #         trial_layout.swap(*node.qargs)

                # check all permutations of CNS subs
                # XXX
                for perm in permutations(cns_sub_candidates):
                    trial_layout = current_layout.copy()
                    for node in perm:
                        trial_layout.swap(*node.qargs)
                    score = self._score_heuristic(
                        "lookahead", front_layer, extended_set, trial_layout
                    )
                    if score < best_score:
                        best_score = score
                        accept_subs = {node: False for node in cns_sub_candidates}
                        for node in perm:
                            accept_subs[node] = True
                # XXX

                # debug: print keys in accept_subs if True
                # print([key for key in accept_subs if accept_subs[key]])
                total_subs += len([key for key in accept_subs if accept_subs[key]])

                # push nodes into execute_gate_list
                for node in cns_sub_candidates:
                    if not accept_subs[node]:
                        outer_execute_gate_list.append(node)
                    else:
                        # convert node to node_prime
                        node_prime = _get_node_cns(node)
                        outer_execute_gate_list.append(node_prime)

            # (I think) this is last layer case, when front_layer is empty
            # just push all nodes into execute_gate_list
            # i.e. using node_prime won't make a difference
            else:
                outer_execute_gate_list += cns_sub_candidates
            # XXX

            if (
                not outer_execute_gate_list
                and len(ops_since_progress) > max_iterations_without_progress
            ):
                raise Warning(
                    "IDK what this condition is, mess up the front_layer logic"
                )
                raise NotImplementedError("needs a debug")
                # Backtrack to the last time we made progress, then greedily insert swaps to route
                # the gate with the smallest distance between its arguments.  This is a release
                # valve for the algorithm to avoid infinite loops only, and should generally not
                # come into play for most circuits.
                self._undo_operations(ops_since_progress, mapped_dag, current_layout)
                self._add_greedy_swaps(
                    front_layer, mapped_dag, current_layout, canonical_register
                )
                continue

            # NOTE, now execute_gate_list has proper node and node_primes
            # push nodes into mapped_dag
            if outer_execute_gate_list:
                for node in outer_execute_gate_list:
                    self._apply_gate(
                        mapped_dag, node, current_layout, canonical_register
                    )

                ops_since_progress = []
                extended_set = None
                outer_execute_gate_list = []

            # debug check, does front_layer only contain 2Q gates?
            for node in front_layer:
                if len(node.qargs) != 2:
                    print("BAD!")

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
                    self.heuristic, front_layer, extended_set, trial_layout, swap_qubits
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

        self.property_set["final_layout"] = current_layout
        # XXX
        self.property_set["mapped_dag_snaps"] = mapped_dag_snaps
        self.property_set["accept_subs"] = total_subs

        if not self.fake_run:
            return mapped_dag

        return dag

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

        # FIXME, I don't know why I have needed to add this check
        # might indicate a larger problem
        # for some reason 1Q gates are getting into front layer
        # breaks this function since len(qargs) != 2

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
