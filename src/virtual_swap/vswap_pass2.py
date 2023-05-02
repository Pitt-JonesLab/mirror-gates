"""Second implementation of virtual-swap routing

Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/simulated_annealing.py

Search problem is initialized by the initial placement of qubits. Important, we want
to decouple the initial placement from the routing problem. If done correctly, we can
use the routing algorithm to assist finding the initial placement (forward-backward pass).

The cost function is the critical path length of the circuit- 
careful to use normalized gate durations.

In example SA algorithm, using neighbors of index for next state.
In our case, we can either (a) move only backwards, (b) move only forwards, or (c) randomly move.

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

The process of propagating gate changes down the DAG breaks assumption that all gates are physical.
This requires an intermediate step between SA iterations to 
1. reconsolidate blocks, 2. enforce routing, 3. recompute critical path cost.
        
"""
from ast import Tuple
import random
import numpy as np
from qiskit.transpiler.passmanager import PassManager
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler.passes.routing import StochasticSwap
from qiskit.transpiler.passes import CountOpsLongestPath
from logging import logging

class VirtualSwap(TransformationPass):
    """Use simulated annealing to route quantum circuit."""
    start_temp = 100
    rate_of_decay = 0.01
    threshold_temp = 1

    def __init__(self, coupling_map, neighbor_func='rand', seed=42, visualize=False):
        """Virtual-swap routing initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            neighbor_func (str): ['rand', 'forward', 'backward'].
            seed (int): Random seed for the stochastic part of the algorithm.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.neighbor_func = neighbor_func
        self.seed = seed
        self.visualize = visualize

    def run(self, dag):
        """Run the VirtualSwapAnnealing pass on `dag`."""
        accepted_dag, accepted_cost = self._cost_cleanup(dag)
        current_temp = self.start_temp
        iterations = 0
        scores = []

        while current_temp > self.threshold_temp:
            working_dag, working_cost = self._SA_iter(accepted_dag)

            if working_cost < accepted_cost:
                accepted_dag = working_dag
                accepted_cost = working_cost

            else:
                # might be backwards
                probability = np.exp((accepted_cost - working_cost) / current_temp)
                logging.debug(f"Probability: {probability}")

                if random.random() < probability:
                    accepted_dag = working_dag
                    accepted_cost = working_cost

            scores.append(accepted_cost)
            iterations += 1
            current_temp *= (1 - self.rate_of_decay)
        
        # visualize scores
        if self.visualize:
            import matplotlib.pyplot as plt
            plt.plot(range(iterations), scores)
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.show()

        return accepted_dag

    def _SA_iter(self, dag):
        """Perform one iteration of the simulated annealing algorithm.

        Args:
            dag (DAGCircuit): DAG to perform SA on.

        Returns:
            Tuple[DAGCircuit, float]: (working_dag, working_cost)
        """
        # pick gate to replace
        sub_node = self.get_random_node(dag)

        # make change in a working copy of the DAG
        working_dag = self._transform_CNS(sub_node, working_dag)

        # compute cost of working_dag
        working_dag, working_cost = self._cost_cleanup(working_dag)
        return working_dag, working_cost
    
    def _get_random_node(self, dag) -> DAGOpNode:
        """Return a node to take SA sub on.
        
        Args:
            dag (DAGCircuit): DAG to pick node from.

        Returns:
            DAGOpNode: Node to take SA sub on.
        """
        # must be (0.5, 0, 0) or (0.5, 0.5, 0) for defined sub rules
        raise NotImplementedError
        if self.neighbor_func == 'rand':
            return self._random_2q_gate(dag)
        elif self.neighbor_func == 'forward':
            return self._random_2q_gate_forward(dag)
        elif self.neighbor_func == 'backward':
            return self._random_2q_gate_backward(dag)

    def _transform_CNS(self, node, dag) -> DAGCircuit:
        """Transform CX into iSWAP+SWAP or iSWAP into CX+SWAP.
        Applies changes to 'dag' in a deep copy. Uses virtual-swap; no real SWAPs added.
        Instead of SWAP, update layout and gate placements.

        Args:
            node_index (int): Index of node to transform.
            dag (DAGCircuit): DAG to transform.
        
        Returns:
            DAGCircuit: Transformed DAG.
        """
        raise NotImplementedError


    def _cost_cleanup(self, dag) -> Tuple[DAGCircuit, float]:
        """Cost function with intermediate steps.
        
        Args:
            dag (DAGCircuit): DAG to compute cost of.
        
        Returns:
            Tuple[DAGCircuit, float]: (dag, cost)
        """
        pass_manager = PassManager()
        pass_manager.append([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])
        
        # NOTE, StochasticSwap not necessarily best choice here.
        pass_manager.append(StochasticSwap(self.coupling_map, seed=self.seed))
        
        # NOTE, using CountOpsLongestPath is valid is every block is 1 cycle.
        pass_manager.append(CountOpsLongestPath())
        
        dag = pass_manager.run(dag)
        cost = pass_manager.property_set['count_ops_longest_path']
        return dag, cost