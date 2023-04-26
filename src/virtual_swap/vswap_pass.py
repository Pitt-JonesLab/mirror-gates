"""Class for virtual-swap routing.

Use :class:qiskit.transpiler.basepasses.TransformationPass.
The idea of virtual-swap is a swap gate that is performed by logical-
physical mapping. Rather than performing a SWAP, the virtual-swap,
vSWAP, relabels the logical qubits, and in effect can be thought of as
SWAP.

Simple strategy: stay in CNOT + SWAP basis. Then, rule is CX has same cost as
Cx+SWAP (CNS). Then, when evaluating mapping, just use CNS=depth 1 rule.

Each loop, choose a random CX, and turn it into CNS. (Then need to recompute routing.)
"""
import random
from typing import Dict
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOpNode
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.passes.routing import StochasticSwap


from copy import deepcopy

class VirtualSwapAnnealing(TransformationPass):
    """Virtual-swap routing."""

    def __init__(self, coupling_map, seed=None):
        """Virtual-swap routing initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            seed (int): Random seed for the stochastic part of the algorithm.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.seed = seed
        # random.seed(self.seed)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the VirtualSwapAnnealing pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.
        """
        dag = self._enforce_routing(dag)
        current_cost = self._cost(dag)
        print(f"Initial cost: {current_cost}")

        # Simulated annealing loop
        for _ in range(20):
            # Make a deep copy of DAGCircuit
            working_dag = deepcopy(dag)

            # Choose a random CX in the DAG
            node = self._random_2q_gate(working_dag)

            # with some probability, SWAP qubits in mapping and reroute instead
            if random.random() < 0.5:
                # Turn CNOT into CNOT + SWAP
                self._transform_CNS(node, working_dag)
            else:
                # update layout, swap 2 random qubits
                layout = self.property_set["layout"]
                qubit1, qubit2 = random.sample(qubits, 2)
                layout.swap(qubit1, qubit2)                
                print(f"Swapped {qubit1} and {qubit2} in mapping")

            # reroute the circuit, must be physically valid before cost
            # this is where I get conceptually confused
            working_dag = self._enforce_routing(working_dag)

            # Calculate cost of the current and new mapping
            new_cost = self._cost(working_dag) 
            from qiskit.converters import dag_to_circuit
            print(dag_to_circuit(working_dag).draw())
            print(f"Current cost: {current_cost}")
            print(f"New cost: {new_cost}")

            # Decide whether to accept the new mapping
            if self._accept(new_cost, current_cost, T=0):
                print("Accepted")
                # Update the current mapping and DAGCircuit
                dag = working_dag
                current_cost = new_cost

            else:
                print("Not accepted")

        print(f"Final cost: {current_cost}")
        return dag

    def _random_2q_gate(self, dag: DAGCircuit) -> DAGOpNode:
        """Choose a random 2-qubit gate in the DAG."""
        two_qubit_nodes = [
            node for node in dag.op_nodes() if node.name == "cx"
        ]
        return random.choice(two_qubit_nodes)

    def _enforce_routing(self, dag):
        """Enforce the routing of the DAGCircuit."""
        layout = self.property_set["layout"]
        router = StochasticSwap(self.coupling_map, trials=20, seed=self.seed, initial_layout=layout)
        return router.run(dag)
    
    def _transform_CNS(self, node: DAGOpNode, dag: DAGCircuit) -> None:
        """Replace a CNOT gate with a CNS gate."""
        cns = QuantumCircuit(2)
        cns.cx(0, 1)
        cns.swap(0, 1)
        cns.swap(0,1)
        cns = circuit_to_dag(cns)

        # Apply the CNS gate
        dag.substitute_node_with_dag(node, cns)
        # I think the substituted node still has successor of the original node
        print(node.qargs)
    
    def _cost(self, dag:DAGCircuit) -> float:
        longest_path = dag.longest_path()[1:-1] # remove input/output nodes
        # only want 2q gates
        longest_path = [node for node in longest_path if len(node.qargs) == 2]
        cost = 0
        index = 0
        while index < len(longest_path):
            if longest_path[index].name == 'cx':
                cost += 2
                index += 1

                # check if successor is a swap on the same qubits
                if index == len(longest_path):
                    break
                if longest_path[index].name == 'swap':
                    if set(longest_path[index].qargs) == set(longest_path[index-1].qargs):
                        index += 1

            elif longest_path[index].name == 'swap':
                # if at end of list, ignore
                if index + 1 == len(longest_path):
                    break
                cost += 3
                index += 1
        return cost
    
    # # FIXME, code uses longest path before edge costs
    # def _cost(self, dag: DAGCircuit, layout_dict: Dict[Qubit, Qubit]) -> float:
    #     """Calculate the cost of the current mapping.
        
    #     Swap gates at the end of the circuit are not counted.
    #     CX = 2, CNS = 2, SWAP = 3
    #     """
    #     longest_path = dag.longest_path()
    #     from qiskit.converters import dag_to_circuit
    #     print(dag_to_circuit(dag).draw())
    #     cost = 0
    #     skip_next = False # tracking for CNS
    #     for node in longest_path:
    #         if not isinstance(node, DAGOpNode):
    #             continue

    #         if node.name == 'cx':
    #             cost += 2
                
    #             # check if successor is a swap on the same qubits
    #             successors = list(dag.successors(node))
    #             if len(successors) == 1 and successors[0].name == 'swap':
    #                 if set(successors[0].qargs) == set(node.qargs):
    #                     skip_next = True
    #                     continue

    #         elif node.name == 'swap' and not skip_next:

    #             # check if SWAP's successor is DAGOutNodes
    #             successors = list(dag.successors(node))
    #             if len(successors) == 1 and isinstance(successors[0], DAGOpNode):
    #                 break
    #             cost += 3
            
    #         skip_next = False
            
    #     return cost


    def _accept(self, new_cost: float, current_cost: float, T:float) -> bool:
        """Decide whether to accept the new mapping."""
        if new_cost <= current_cost:
            return True
        else:
            # simulated annealing, P(accept) = exp(-delta_cost / T)
            prob = np.exp(-(new_cost - current_cost) / T)
            print(f"Probability of accepting: {prob}")
            return random.random() < prob
            
    # def _apply_virtual_swap(
    #         self, dag: DAGCircuit, node: DAGNode, layout_dict: Dict[Qubit, Qubit]
    # ) -> None:
    #     """Apply a virtual swap.
        
    #     Rather than messing with the DAG and qubit layouts, we simply apply a SWAP gate, 
    #     importantly, this SWAP should not be counted in the depth of the circuit.

    #     After injecting the SWAP gate, need to recompute routing. We should not allow the case
    #     where the SWAP gate gets immediately undone by another SWAP gate.
    #     """
    #     # find the node, split the circuit into two parts, and insert the SWAP gate
    #     descendants = dag.descendants(node)
    #     for n in descendants:
    #         if isinstance(n, DAGOpNode):
    #             dag.remove_op_node(n)


    def _apply_virtual_swap(
        self, dag: DAGCircuit, node: DAGOpNode, layout_dict: Dict[Qubit, Qubit]
    ) -> None:
        """Apply a virtual-swap.

        Apply a virtual-swap at the given node in the DAG and update the layout.

        Args:
            dag (DAGCircuit): DAG to map.
            node (DAGNode): Node at which to apply the virtual-swap.
            layout_dict (Dict[Qubit, Qubit]): Current layout of qubits.
        """
        if len(node.qargs) != 2:
            return

        # Update the layout dictionary
        layout_dict[node.qargs[0]], layout_dict[node.qargs[1]] = (
            layout_dict[node.qargs[1]],
            layout_dict[node.qargs[0]],
        )

        # Propagate the changes through the remaining gates in the DAG
        for successor_node in dag.successors(node):
            if successor_node.type == "op":
                new_qargs = [layout_dict[q] for q in successor_node.qargs]
                successor_node.qargs = new_qargs
            
        
