"""Second implementation of virtual-swap routing.

Ref: https://github.com/TheAlgorithms/Python/blob/master/searches/simulated_annealing.py

Search problem is initialized by the initial placement of qubits. Important, we want
to decouple the initial placement from the routing problem. If done correctly, we can
use routing algorithm to assist finding the initial placement (forward-backward pass).

The cost function is the critical path length of the circuit-
careful to use normalized gate durations.

In example SA algorithm, using neighbors of index for next state.
We can either (a) move only backwards, (b) move only forwards, or (c) randomly move.

Assume given a circuit with only CX gates. Valid moves are:
    (a) decompose CX gate into iSWAP + SWAP
    (b) decompose CX gate into sqiSWAP + sqiSWAP
    _____________
    (c) decompose iSWAP into CX + SWAP
    (d) decompose sqiSWAP + sqiSWAP into CX

Unclear to me yet best way to keep track of intermediate representations.
On one hand, we prefer to keep everything as CX gates.
Better, is in terms of (c1,c2,c3) coordinates.
However, with coordinates, don't know if sub is (a) or (b) since equivalent.
Requires consolidate blocks, which is a good idea anyway.

Alternatively, rather than placing a SWAP in subs (a) and (c); instead,
just update the DAG by propagating changes on gates on updated layout.
I like this method because the SWAP gate is not counted in the depth of the circuit.

So each 2Q gate will either be (0.5, 0, 0) or (.5, .5, 0) in (c1, c2, c3) coordinates.
We can interchange between them by applying a virtual-swap.

Requires real-SWAP gates throughout circuit for routing. We can't sub these;
however, the SWAPs should be part of the consolidate blocks pass.

Finish by making any remaining CX (0.5, 0, 0) use the sqiSWAP + sqiSWAP decomposition.

Edge cases:
    1. Coordinate shows up we don't know how to handle
        - not sure yet, need to see an example. We can just decompose into sqiSWAP.

Propagating gate changes down DAG breaks assumption that gates are physical.
This requires an intermediate step between SA iterations to
1. reconsolidate blocks, 2. enforce routing, 3. recompute critical path cost.
"""
import logging
import random
from copy import deepcopy
from typing import Tuple

import numpy as np
from qiskit.circuit.library import CXGate, iSwapGate
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import (
    Collect2qBlocks,
    ConsolidateBlocks,
    CountOpsLongestPath,
)
from qiskit.transpiler.passes.routing import BasicSwap
from weylchamber import c1c2c3

logger = logging.getLogger("VSWAP")

# can be overriden in __init__, placed here for convenience
default_start_temp = 1
default_rate_of_decay = 0.001
# BAD BAD BAD, should be below 1, makes sure gets to do greedy part of algorithm
default_threshold_temp = 0.01


class VirtualSwap(TransformationPass):
    """Use simulated annealing to route quantum circuit.

    Assumes input circuit is written in terms of CX gates. Output
    circuit is written in terms of sqiSWAP gates.
    """

    def __init__(
        self,
        coupling_map,
        neighbor_func="rand",
        seed=None,
        return_best=True,
        sa_params=(
            default_start_temp,
            default_rate_of_decay,
            default_threshold_temp,
        ),
        visualize=False,
    ):
        """Virtual-swap routing initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            neighbor_func (str): ['rand', 'forward', 'backward'].
            seed (int): Random seed for the stochastic part of the algorithm.
            return_best (bool): If True, return the best DAG instead of the last DAG.
            sa_params (tuple): (start_temp, rate_of_decay, threshold_temp)
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.neighbor_func = neighbor_func
        self.node_index = 0
        if self.neighbor_func == "backward":
            self.node_index = -1
        self.seed = seed
        self.return_best = return_best
        self.start_temp, self.rate_of_decay, self.threshold_temp = sa_params
        self.visualize = visualize

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the VirtualSwapAnnealing pass on `dag`."""
        logger.debug(f"Initial:\n{dag_to_circuit(dag).draw(fold=-1)}")

        accepted_dag, accepted_cost = self._cost_cleanup(dag)
        logger.debug(f"Initial:\n{dag_to_circuit(accepted_dag).draw(fold=-1)}")

        best_dag = None
        best_cost = None
        current_temp = self.start_temp
        iterations = 0
        scores = []

        while current_temp > self.threshold_temp:
            working_dag, working_cost = self._SA_iter(accepted_dag)
            logger.debug(f"Working:\n{dag_to_circuit(working_dag).draw(fold=-1)}")
            logger.info(f"Working: {working_cost}")

            if best_cost is None or working_cost < best_cost:
                best_dag = working_dag
                best_cost = working_cost
                logger.info(f"Best: {best_cost}")

            if self._SA_accept(working_cost, accepted_cost, current_temp):
                accepted_dag = working_dag
                accepted_cost = working_cost
                logger.info(f"Accepted: {accepted_cost}")
            else:
                logger.info(f"Rejected: {working_cost}")

            scores.append(accepted_cost)
            iterations += 1
            current_temp *= 1 - self.rate_of_decay

        # visualize scores
        if self.visualize:
            import matplotlib.pyplot as plt

            plt.plot(range(iterations), scores)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.show()
        self.property_set["scores"] = scores

        if self.return_best:
            return best_dag
        return accepted_dag

    def _SA_iter(self, dag: DAGCircuit):
        """Perform one iteration of the simulated annealing algorithm.

        Args:
            dag (DAGCircuit): DAG to perform SA on.

        Returns:
            Tuple[DAGCircuit, float]: (working_dag, working_cost)
        """
        working_dag = deepcopy(dag)

        # pick gate to replace
        sub_node = self._get_next_node(working_dag)

        # make change in a working copy of the DAG
        working_dag = self._transform_CNS(sub_node, working_dag)

        # compute cost of working_dag
        working_dag, working_cost = self._cost_cleanup(working_dag)
        return working_dag, working_cost

    def _SA_accept(self, working_cost, accepted_cost, current_temp) -> bool:
        """Return True if we should accept the working state."""
        if working_cost < accepted_cost:
            return True
        else:
            probability = np.exp((accepted_cost - working_cost) / current_temp)
            logger.info(f"Probability: {probability}")
            return random.random() < probability

    def _get_next_node(self, dag: DAGCircuit) -> DAGOpNode:
        """Return a node to take SA sub on.

        Args:
            dag (DAGCircuit): DAG to pick node from.

        Returns:
            DAGOpNode: Node to take SA sub on.
        """
        # must be (0.5, 0, 0) or (0.5, 0.5, 0) for defined sub rules
        # its likely that some consolidation gives (x, 0, 0)
        # let's just mke sure its not very often
        # Ensure the sub rules are (0.5, 0, 0) or (0.5, 0.5, 0),
        # and consolidation gives (x, 0, 0)
        ###########################
        valid_sub_rules = {(0.5, 0, 0), (0.5, 0.5, 0)}
        # valid_consolidations = valid_sub_rules | {(0.5, 0.5, 0.5)}
        # l1 = [c1c2c3(np.array(sel.op)) for sel in dag.op_nodes()]
        # count1 = sum(el in valid_sub_rules for el in l1)
        # count2 = sum(el in valid_consolidations for el in l1)
        # count3 = len(dag.op_nodes())
        # logger.info(f"CX+iSWAP count: {count1}")
        # logger.info(f"CX+iSWAP+SWAP count: {count2}")
        # logger.info(f"Total count: {count3}")
        # ############################

        selected_node = None
        while (
            selected_node is None
            or c1c2c3(np.array(selected_node.op)) not in valid_sub_rules
        ):
            if self.neighbor_func == "rand":
                selected_node = random.choice(dag.op_nodes())
            # use node index to select next node in valid_sub_rules
            # if index out of bounds, reset
            elif self.neighbor_func == "forward":
                selected_node = dag.op_nodes()[self.node_index]
                self.node_index += 1
                if self.node_index >= len(dag.op_nodes()):
                    self.node_index = 0
            # reset goes to end of list
            elif self.neighbor_func == "backward":
                selected_node = dag.op_nodes()[self.node_index]
                self.node_index -= 1
                if self.node_index < 0:
                    self.node_index = len(dag.op_nodes()) - 1

        assert len(selected_node.qargs) == 2
        assert c1c2c3(np.array(selected_node.op)) in [(0.5, 0, 0), (0.5, 0.5, 0)]
        return selected_node

    def _transform_CNS(self, node: DAGOpNode, dag: DAGCircuit) -> DAGCircuit:
        """Transform CX into iSWAP+SWAP or iSWAP into CX+SWAP.

        Applies changes to 'dag' in-place. Uses virtual-swap; no real SWAPs added.
        Instead of SWAP, update layout and gate placements.

        Args:
            node_index (int): Index of node to transform.
            dag (DAGCircuit): DAG to transform.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        # 1. transform node in temp_dag
        # XXX will need to define the sub rules more exactly
        coord = c1c2c3(np.array(node.op))
        if coord == (0.5, 0, 0):
            # transform CX into iSWAP
            # XXX not preserving unitary correctness
            dag.substitute_node(node, iSwapGate(), inplace=True)
        elif coord == (0.5, 0.5, 0):
            # transform iSWAP into CX
            # XXX not preserving unitary correctness
            dag.substitute_node(node, CXGate(), inplace=True)
        else:
            raise ValueError(f"Invalid node: {node}")

        # 2. propagate changes down the DAG
        # NOTE there is definitely a better way to do this
        # find clever use of dag.substitute_node_with_dag

        # placement of the virtual-swap gate
        # copy maybe not necessary, keep for safety :)
        working_layout = self.property_set["layout"].copy()
        working_layout.swap(node.qargs[0], node.qargs[1])

        # update gate-wire placement
        for successor_node in dag.descendants(node):
            if isinstance(successor_node, DAGOpNode):
                successor_node.qargs = [
                    working_layout[qarg.index] for qarg in successor_node.qargs
                ]

        return dag

    def _cost_cleanup(self, dag: DAGCircuit) -> Tuple[DAGCircuit, float]:
        """Cost function with intermediate steps.

        Args:
            dag (DAGCircuit): DAG to compute cost of.

        Returns:
            Tuple[DAGCircuit, float]: (dag, cost)
        """
        # NOTE potenially very bad to have nested transpiler passes
        # https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/transpiler/passes/layout/sabre_layout.py#L345
        # use the property_set to pass information between passes
        # avoids overhead of converting to/from DAGCircuit

        # swap_pass = StochasticSwap(self.coupling_map, seed=self.seed)
        # swap_pass.property_set = self.property_set
        # dag = swap_pass.run(dag)

        # XXX could be the main bottleneck,
        # do we really need this, to evaluate if the replacement is good,
        # we should be able to decide purely based on topology,
        # does it make the outputs closer to their next qubit
        # TODO
        swap_pass = BasicSwap(self.coupling_map)
        swap_pass.property_set = self.property_set
        dag = swap_pass.run(dag)

        routed_qc = dag_to_circuit(dag)
        logger.debug(f"Routed:\n{routed_qc.draw(fold=-1)}")

        collect_pass = Collect2qBlocks()
        collect_pass.property_set = swap_pass.property_set
        dag = collect_pass.run(dag)

        consolidate_pass = ConsolidateBlocks(force_consolidate=True)
        consolidate_pass.property_set = collect_pass.property_set
        dag = consolidate_pass.run(dag)

        # XXX this is currently broken because of the way consolidate pass works
        # better would be to check for each unitary, its decomposition cost
        cost_pass = CountOpsLongestPath()
        cost_pass.property_set = consolidate_pass.property_set
        cost_pass.run(dag)  # doesn't return anything, just updates property_set

        cost_dict = cost_pass.property_set["count_ops_longest_path"]
        cost = sum(
            3 * val if key == "swap" else 2 * val for key, val in cost_dict.items()
        )
        return dag, cost
