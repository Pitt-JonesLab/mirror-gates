"""CNS Transformations for mirror gates."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.extensions import UnitaryGate

from mirror_gates.fast_unitary import FastConsolidateBlocks, NoCheckUnitary

# Global CNS Transformations
# cx -> iswap
cx_replace = QuantumCircuit(2, 0, name="iswap_prime")
cx_replace.h(1)
cx_replace.rz(-np.pi / 2, 0)
cx_replace.rz(-np.pi / 2, 1)
cx_replace.iswap(0, 1)
# cx_replace.append(SiSwapGate(), [0, 1])
# cx_replace.append(SiSwapGate(), [0, 1])
cx_replace.h(0)

# iswap -> cx
iswap_replace = QuantumCircuit(2, 0, name="cx_prime")
iswap_replace.rz(np.pi / 2, 0)
iswap_replace.rz(np.pi / 2, 1)
iswap_replace.h(1)
iswap_replace.cx(0, 1)
iswap_replace.h(1)


def _get_node_cns(node: DAGOpNode, use_fast_settings: bool = True) -> Instruction:
    """Get the CNS transformation for a given node."""
    if len(node.qargs) != 2:
        raise ValueError("Only supports 2Q gates")

    # NOTE, the UnitaryGate() constructor is a bit expensive
    if use_fast_settings:
        new_op = SwapGate().to_matrix() @ node.op.to_matrix()
        new_unitary = NoCheckUnitary(new_op, label="u+swap")

        # TODO: calculate mirror coordinate directly
        # from monodromy.coordinates import mirror_monodromy_coordinate
        # _monodromy_coord = mirror_monodromy_coordinate(node.op._monodromy_coord)
        new_unitary._monodromy_coord = FastConsolidateBlocks.unitary_to_coordinate(
            new_unitary
        )
    else:
        new_op = SwapGate().to_matrix() @ node.op.to_matrix()
        new_unitary = UnitaryGate(new_op, label="u+swap")

    return DAGOpNode(op=new_unitary, qargs=node.qargs)


def cns_transform(dag: DAGCircuit, *h_nodes, preserve_layout=False) -> DAGCircuit:
    """Transform DAG by applying CNS transformations on multiple nodes.

    Args:
        dag (DAGCircuit): DAG to be transformed (will not be modified)
        h_nodes (DAGOpNode): Nodes to be transformed.
        preserve_layout (bool): If True, the layout of the original DAG is preserved.
        Use this option if testing equivalence of original and transformed circuits.
    """
    new_dag = dag.copy_empty_like()

    if preserve_layout:  # convert so can be modified
        h_nodes = list(h_nodes)

    # Initialize layout for each node
    layout = {qarg: qarg for node in dag.topological_op_nodes() for qarg in node.qargs}

    for node in dag.topological_op_nodes():
        qargs = [layout.get(qarg, qarg) for qarg in node.qargs]

        # check if node is in list of nodes to be transformed
        # FIXME, this is true multiple times,
        # semantic_eq checks if is a CX but not if the exact same CX
        if any(node == h_node for h_node in h_nodes):
            # if any(DAGNode.semantic_eq(node, h_node) for h_node in h_nodes):
            try:  # checks if has a defined CNS transformation
                node_prime = _get_node_cns(node)
                new_dag.apply_operation_back(node_prime.op, qargs)
                # swap values in layout
                layout[node.qargs[0]], layout[node.qargs[1]] = qargs[1], qargs[0]
            except ValueError:
                new_dag.apply_operation_back(node.op, qargs)
                h_nodes.remove(node)
        else:
            new_dag.apply_operation_back(node.op, qargs)

    if preserve_layout:
        for h_node in reversed(h_nodes):
            new_dag.apply_operation_back(SwapGate(), h_node.qargs)

    return new_dag


# legacy code, only works for a single node
# def cns_transform(dag: DAGCircuit, h_node, preserve_layout=False):
#     """Alternative implementation, adds nodes into blank copy of dag."""
#     new_dag = dag.copy_empty_like()

#     flip_flag = False

#     swap_wires = {
#         qarg1: qarg2 for qarg1, qarg2 in zip(h_node.qargs, h_node.qargs[::-1])
#     }

#     for node in dag.topological_op_nodes():
#         # if node == h_node:
#         if DAGNode.semantic_eq(node, h_node):
#             # here we add the cns transformation, and use the flip flag
#             # flip_flag tells us from this gate onwards, qargs will reverse
#             # effectively, we are adding the virtual swap here
#             new_dag.apply_operation_back(_get_node_cns(node).op, node.qargs)
#             flip_flag = True

#         else:
#             if flip_flag:
#                 new_dag.apply_operation_back(
#                     node.op, [swap_wires.get(qarg, qarg) for qarg in node.qargs]
#                 )
#             else:
#                 new_dag.apply_operation_back(node.op, node.qargs)

#     # fix with a swap
#     # if preserve_layout:
#       # new_dag.apply_operation_back(SwapGate(), h_node.qargs)
#     return new_dag
