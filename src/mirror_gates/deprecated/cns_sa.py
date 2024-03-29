"""V4 Virtual-SWAP Layout+Routing Pass.

The main thing that we need to take careful consideration of is the
virtual-physical Layout objects. I think I have been mixing up the
indices of which is which. Also, I might prefer to use a custom DAG
class internal to this pass. This would let me edit attributes of the
OpNodes.

Second, we need to focus on preserving the unitary operation. This means
that we should be able to verify that the circuit before and after
transformation is still the same unitary.

Third, consider a version that makes the CNS substitution each
iteration, this makes it easier to visualize what changes are being
made. But also, we can make a version that waits until the end. Swapping
the wires of gate descendants is expensive, and we don't need to do it
in order to calculate topological distance (can just reference a dynamic
layout). (Third, ideally this should be addressed by a more efficient
custom DAG operation. Ref:
https://github.com/Qiskit/qiskit-terra/pull/9863)
: https: //github.com/Qiskit/qiskit-terra/pull/9863)
"""

import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import Optimize1qGates, Unroller

from mirror_gates.cns_transform import cns_transform

# for debuging

# from qiskit.transpiler.passes import Layout2qDistance
# from qiskit.circuit.library import SwapGate

# can be overriden in __init__, placed here for convenience
default_start_temp = 8
default_rate_of_decay = 0.01
default_threshold_temp = 0.1


class VSwapPass(TransformationPass):
    """Use simulated annealing to choose between CNS decompositions."""

    def __init__(
        self,
        coupling_map,
        start_temp=default_start_temp,
        rate_of_decay=default_rate_of_decay,
        threshold_temp=default_threshold_temp,
        visualize=False,
        seed=None,
    ):
        """Initialize the VSwapPass."""
        super().__init__()
        self.basis = ["u", "cx", "iswap"]
        self.requires.append(Unroller(self.basis))
        self.coupling_map = coupling_map
        self.start_temp = start_temp
        self.rate_of_decay = rate_of_decay
        self.threshold_temp = threshold_temp
        self.visualize = visualize
        random.seed(seed)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the VSwapPass."""
        # setup
        self.probabilities = []
        self.cost_list = []
        self.dag = deepcopy(dag)  # dag.copy()
        self.cost = self._evaluate_cost(self.dag)
        self.temp = self.start_temp

        # tracking winners
        best_dag = None
        best_cost = np.inf

        # main loop
        while self.temp > self.threshold_temp:
            self._SA_iter()
            self.cost_list.append(self.cost)
            if self.cost < best_cost:
                best_dag = deepcopy(self.dag)  # self.dag.copy()
                best_cost = self.cost
            self.temp *= 1 - self.rate_of_decay

        if self.visualize:
            self._visualize()

        return best_dag

    def _visualize(self) -> None:
        """Plot costs and probabilities vs.

        iteration.
        """
        plt.plot(self.cost_list)
        plt.title("Cost vs. Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        ax2 = plt.twinx()
        ax2.plot(self.probabilities, color="red", marker=".", alpha=0.25)
        ax2.set_ylabel("Probability", color="red")
        plt.show()

    def _SA_iter(self) -> None:
        """Perform a simulated annealing iteration."""
        # FIXME, for speed-up, can make a copy only after accepting a change
        new_dag = deepcopy(self.dag)  # self.dag.copy()

        sub_node = self._get_next_node(new_dag)

        # make CNS transformation
        new_dag = cns_transform(new_dag, sub_node)

        # XXX move into new function?
        # cleanup using unroller and 1Q smush
        # the cns_transform, leaves dag with an op node not in terms of iswap/cx
        # this is because we sub in a circuit as an instruction, not a gate
        # therefore, use this unroller to convert back into gate terms
        # not sure if there is a better way, seems to breaking the DAG
        # FIXME
        unroller = Unroller(self.basis)
        unroller.property_set = self.property_set
        new_dag = unroller.run(new_dag)
        # NOTE, the 1Q gate smush is not necessary,
        # because cost function only over 2Q gates
        # makes circuits easier to read when debugging
        collect_1q = Optimize1qGates(self.basis)
        collect_1q.property_set = unroller.property_set
        new_dag = collect_1q.run(new_dag)
        self.property_set = collect_1q.property_set

        # evaluate cost
        new_cost = self._evaluate_cost(new_dag)

        # accept or reject
        if self._SA_accept(new_cost):
            self.dag = new_dag
            self.cost = new_cost
            # XXX debug
            # from qiskit.converters import dag_to_circuit
            # print(dag_to_circuit(self.dag).draw(fold=-1))

    def _SA_accept(self, new_cost) -> bool:
        """Calculate acceptance probability."""
        if new_cost < self.cost:
            self.probabilities.append(1)
            return True
        else:
            prob = np.exp((self.cost - new_cost) / self.temp)
            self.probabilities.append(prob)
            return prob > np.random.random()

    def _get_next_node(self, dag: DAGCircuit) -> DAGOpNode:
        """Get the next node to perform the CNS transformation on."""
        selected_node = None

        valid_subs = ["cx", "iswap"]

        while True:
            if selected_node is None:
                # random choice over gates
                selected_node = np.random.choice(dag.gate_nodes())

            # must be a two-qubit gate
            # FIXME can remove this, just by checking if 'cx' in name
            if len(selected_node.qargs) != 2:
                selected_node = None
                continue

            if selected_node.name not in valid_subs:
                selected_node = None
                continue
            break
        return selected_node

    def _evaluate_cost(self, dag: DAGCircuit) -> float:
        """Evaluate the cost of the current dag.

        We want cost function to be closely related to
        depth/parallelism. Therefore, need to define a longest path with
        respect to some weight function. DAG longest path considers all
        nodes equivalent duration.

        Moreover, we cannot induce SWAP gates between every SA
        iteration. Therefore, this cost is a heuristic based on
        topological distance.
        """
        # FIXME

        # method 1, use depth :(
        return dag.depth()

        ## NOTE ApplyLayout changes qargs into physical indices
        # don't do this conversion, because already physical :)
        # v_to_p = self.property_set["layout"].get_virtual_bits()

        # method 2, use topological distance over all gates
        # is an okay heuristic, but is related to total gates
        # instead we want critical path gates
        # two_qubit_ops = list(dag.two_qubit_ops())
        # total_cost = len(two_qubit_ops)
        # total_distance = 0
        # for node in two_qubit_ops:
        #     # get physical index of node qargs
        #     phys_qargs = [arg.index for arg in node.qargs]

        #     # get topological distance
        #     total_distance += self.coupling_map.distance(*phys_qargs) - 1
        # total_cost += total_distance
        # return total_cost

        # def weight_fn(_1, target_node, _2):
        #     nonlocal cost_dag
        #     nonlocal cost_temp_layout
        #     # source_node = cost_dag._multi_graph[source_node]
        #     target_node = cost_dag._multi_graph[target_node]
        #     if not isinstance(target_node, DAGOpNode):
        #         return 0
        #     if len(target_node.qargs) != 2:
        #         return 0
        #     arg1 = cost_temp_layout.get_virtual_bits()[target_node.qargs[0]]
        #     arg2 = cost_temp_layout.get_virtual_bits()[target_node.qargs[1]]
        #     return self.coupling_map.distance(arg1, arg2)
