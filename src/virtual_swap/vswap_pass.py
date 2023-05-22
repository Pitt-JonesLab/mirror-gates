"""V4 Virtual-SWAP Layout+Routing Pass.

The main thing that we need to take careful consideration of is
the virtual-physical Layout objects. I think I have been mixing
up the indices of which is which. Also, I might prefer to use
a custom DAG class internal to this pass. This would let me edit
attributes of the OpNodes.

Second, we need to focus on preserving the unitary operation.
This means that we should be able to verify that the circuit
before and after transformation is still the same unitary.

Third, consider a version that makes the CNS substitution each iteration,
this makes it easier to visualize what changes are being made. But also,
we can make a version that waits until the end. Swapping the wires of
gate descendants is expensive, and we don't need to do it in order to
calculate topological distance (can just reference a dynamic layout).
(Third, ideally this should be addressed by a more efficient custom DAG operation.
Ref: https://github.com/Qiskit/qiskit-terra/pull/9863)
"""

import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import Optimize1qGates, Unroller

# from qiskit.transpiler.passes import Layout2qDistance
# from qiskit.circuit.library import SwapGate

# Global CNS Transformations. TODO move to _equiv.py
# cx -> iswap
cx_replace = QuantumCircuit(2, 0)
cx_replace.h(1)
cx_replace.rz(-np.pi / 2, 0)
cx_replace.rz(-np.pi / 2, 1)
cx_replace.iswap(0, 1)
cx_replace.h(0)
cx_replace.draw("mpl")

# iswap -> cx
iswap_replace = QuantumCircuit(2, 0)
iswap_replace.rz(np.pi / 2, 0)
iswap_replace.rz(np.pi / 2, 1)
iswap_replace.h(1)
iswap_replace.cx(0, 1)
iswap_replace.h(1)

# can be overriden in __init__, placed here for convenience
default_start_temp = 5
default_rate_of_decay = 0.01
default_threshold_temp = 0.5


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
        ax2.plot(self.probabilities, color="red", marker=".")
        ax2.set_ylabel("Probability", color="red")
        plt.show()

    def _SA_iter(self) -> None:
        """Perform a simulated annealing iteration."""
        sub_node = self._get_next_node()

        # FIXME, for speed-up, can make a copy only after accepting a change
        new_dag = deepcopy(self.dag)  # self.dag.copy()

        # make CNS transformation
        self._cns_transform(new_dag, sub_node)

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

    def _get_next_node(self) -> DAGOpNode:
        """Get the next node to perform the CNS transformation on."""
        selected_node = None

        valid_subs = ["cx", "iswap"]

        while True:
            if selected_node is None:
                # random choice over gates
                selected_node = np.random.choice(self.dag.gate_nodes())

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

    def _cns_transform(self, dag, h_node) -> None:
        """Perform the CNS transformation on the given node.

        Modifies the DAG in place, thus should be called on a copy.
        """
        swap_wires = {
            qarg1: qarg2 for qarg1, qarg2 in zip(h_node.qargs, h_node.qargs[::-1])
        }

        # NOTE, include the node itself, we want to reference set of all outgoing edges
        node_list = list(dag.descendants(h_node))
        node_list.insert(0, h_node)

        for source in node_list:
            # first, update edge arguments
            for target in dag.successors(source):
                edge = dag._multi_graph.get_edge_data(source._node_id, target._node_id)
                dag._multi_graph.update_edge(
                    source._node_id, target._node_id, swap_wires.get(edge, edge)
                )

            # second, update the node's qargs
            if source == h_node:
                # here replace the operation...
                if source.name == "cx":
                    dag.substitute_node(source, cx_replace.to_instruction())
                elif source.name == "iswap":
                    dag.substitute_node(source, iswap_replace.to_instruction())
                else:
                    raise ValueError("Unsupported operation")

                # source doesn't get wire swap applied on itself
                continue
            elif isinstance(source, DAGOpNode):
                source.qargs = [swap_wires.get(qarg, qarg) for qarg in source.qargs]

            # third, update output nodes
            if isinstance(source, DAGOutNode):
                source.wire = swap_wires.get(source.wire, source.wire)

        # add a swap a swap at end of circuit
        # NOTE, preserves unitary for equivalence checks
        # final SWAP gates can be removed by the optimizer
        # dag.apply_operation_back(SwapGate(), h_node.qargs)
        # FIXME update dag output map, see optimizeswap pass
        dag.output_map.update(swap_wires)

        return dag

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
        # return dag.depth()

        v_to_p = self.property_set["layout"].get_virtual_bits()
        # method 2, use topological distance over all gates
        # is an okay heuristic, but is related to total gates
        # instead we want critical path gates
        two_qubit_ops = list(dag.two_qubit_ops())
        total_cost = len(two_qubit_ops)
        total_distance = 0
        for node in two_qubit_ops:
            # get physical index of node qargs
            phys_qargs = [v_to_p[qarg] for qarg in node.qargs]
            # get topological distance
            total_distance += self.coupling_map.distance(*phys_qargs) - 1
        total_cost += total_distance
        return total_cost

        # method 3, use topological distance
        # # but also need a notion of consolidated block decomp cost
        # two_qubit_ops = list(dag.two_qubit_ops())
        # total_cost = 0
        # total_distance = 0
        # for node in two_qubit_ops:
        #     # get physical index of node qargs
        #     phys_qargs = [v_to_p[qarg] for qarg in node.qargs]
        #     # get topological distance
        #     total_distance += self.coupling_map.distance(*phys_qargs) - 1
        # total_cost += total_distance
        # return total_cost

        # distance_pass = Layout2qDistance(self.coupling_map, "distance_score")
        # distance_pass.property_set = self.property_set
        # distance_pass.run(dag)
        # return self.property_set["distance_score"]

    # use built-in method
