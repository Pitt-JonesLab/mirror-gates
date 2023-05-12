"""Third implementation of virtual-swap routing.

I think we need to greatly simplify the problem. We are overcomplicating
things by trying to do decomposition at same time, which gives us a huge
time complexity cost in having to search the dag with replacements, and
do block consolidation.

Let's boil this pass down to what it really does. Every CNOT gate can
either have its output normal or swapped.

Later, we can choose to decompose differently the normal vs swapped
output cases. But here, we just need to figure out which of the CNOTs we
would like to flip.

Once I have basic implemented, I think this needs to incorporate
lookahead.
"""
import logging
import random
from copy import deepcopy
from typing import Tuple

import numpy as np
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from weylchamber import c1c2c3

logger = logging.getLogger("VSWAP")

# can be overriden in __init__, placed here for convenience
default_start_temp = 5
default_rate_of_decay = 0.01
# BAD BAD BAD, should be below 1, makes sure gets to do greedy part of algorithm
default_threshold_temp = 0.1


class VirtualSwap(TransformationPass):
    """Use simulated annealing to route quantum circuit.

    Assumes input circuit is written in terms of CX gates. Output labels
    each CX has having an output that is either normal or swapped.
    Later, we can choose different decompositions for normal vs swapped
    cases.
    """

    def __init__(
        self,
        coupling_map,
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
            seed (int): Random seed for the stochastic part of the algorithm.
            return_best (bool): If True, return the best DAG instead of the last DAG.
            sa_params (tuple): (start_temp, rate_of_decay, threshold_temp)
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.seed = seed
        self.return_best = return_best
        self.start_temp, self.rate_of_decay, self.threshold_temp = sa_params
        self.visualize = visualize
        self.probabilities = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the VirtualSwapAnnealing pass on `dag`."""
        # logger.debug(f"Initial:\n{dag_to_circuit(dag).draw(fold=-1)}")

        # initialize accepted state
        accepted_dag, accepted_cost = self._cost_cleanup(dag)
        accepted_copy = deepcopy(accepted_dag)

        # logger.debug(f"Initial:\n{dag_to_circuit(accepted_dag).draw(fold=-1)}")

        best_dag = None
        best_cost = None
        current_temp = self.start_temp
        iterations = 0
        scores = []
        self.probabilities = []

        while current_temp > self.threshold_temp:
            working_dag, working_cost = self._SA_iter(accepted_copy)
            # logger.debug(f"Working:\n{dag_to_circuit(working_dag).draw(fold=-1)}")
            logger.info(f"Working: {working_cost}")

            if best_cost is None or working_cost < best_cost:
                best_dag = working_dag
                best_cost = working_cost
                logger.info(f"Best: {best_cost}")

            if self._SA_accept(working_cost, accepted_cost, current_temp):
                accepted_dag = working_dag
                # NOTE, copy here so ends up calling deepcopy less often
                accepted_copy = deepcopy(accepted_dag)
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
            plt.title("Simulated Annealing")
            # plot probabilities with separate y-axis
            ax2 = plt.twinx()
            ax2.scatter(range(iterations), self.probabilities, color="red", marker=".")
            ax2.set_ylabel("Probability")
            plt.show()
        self.property_set["scores"] = scores

        if self.return_best:
            return best_dag
        return accepted_dag

    def _SA_iter(self, working_dag: DAGCircuit):
        """Perform one iteration of the simulated annealing algorithm.

        Args:
            working_dag (DAGCircuit): DAG to perform SA on. This DAG will be modified.
            Therefore, pass in a deepcopy. Create a copy only when accepting changes,
            this means can call deepcopy less often.

        Returns:
            Tuple[DAGCircuit, float]: (working_dag, working_cost)
        """
        # changed so deepcopy is handled outside of this function
        # working_dag = deepcopy(dag)

        # mark next gate to change output type
        _ = self._get_next_node(working_dag)

        # compute cost of working_dag
        working_dag, working_cost = self._cost_cleanup(working_dag)
        return working_dag, working_cost

    def _SA_accept(self, working_cost, accepted_cost, current_temp) -> bool:
        """Return True if we should accept the working state."""
        if working_cost < accepted_cost:
            self.probabilities.append(1)
            return True
        else:
            probability = np.exp((accepted_cost - working_cost) / current_temp)
            logger.info(f"Probability: {probability}")
            self.probabilities.append(probability)
            return random.random() < probability

    def _get_next_node(self, dag: DAGCircuit) -> DAGOpNode:
        """Mark a new node to take SA sub on.

        Args:
            dag (DAGCircuit): DAG to pick node from.

        Returns:
            DAGOpNode: Node to take SA sub on.
        """
        selected_node = None
        while selected_node is None or selected_node.op.name not in ["cx", "cx_m"]:
            selected_node = random.choice(list(dag.two_qubit_ops()))

        assert len(selected_node.qargs) == 2
        assert c1c2c3(np.array(selected_node.op)) in [(0.5, 0, 0)]

        # mark node as change output type
        if selected_node.op.name == "cx":
            selected_node.op.name = "cx_m"
        elif selected_node.op.name == "cx_m":
            selected_node.op.name = "cx"

        return selected_node

    def _cost_cleanup(self, dag: DAGCircuit) -> Tuple[DAGCircuit, float]:
        """Cost function with intermediate steps.

        Args:
            dag (DAGCircuit): DAG to compute cost of.

        Returns:
            Tuple[DAGCircuit, float]: (dag, cost)
        """

        # we need this in order to get bit indexes
        # FIXME, to save time, can only recompute this when we change the DAg
