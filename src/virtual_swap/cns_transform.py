"""CNS Transformations for Virtual Swap."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOpNode

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


def _get_node_cns(node: DAGOpNode):
    """Get the CNS transformation for a given node."""
    if node.name == "cx":
        # return cx_replace.to_instruction()
        ret_node = DAGOpNode(op=cx_replace.to_instruction(), qargs=node.qargs)
        if node.op.definition is not None:
            print(node.op.definition.global_phase)
            ret_node.op.definition.global_phase = node.op.definition.global_phase
        return ret_node
    elif node.name == "iswap":
        # return iswap_replace.to_instruction()
        return DAGOpNode(op=iswap_replace.to_instruction(), qargs=node.qargs)
    else:
        raise ValueError(f"Unsupported operation, {node.name}")


def _cns_transform(dag: DAGCircuit, h_node):
    """Alternative implementation, adds nodes into blank copy of dag."""
    new_dag = dag.copy_empty_like()

    flip_flag = False

    swap_wires = {
        qarg1: qarg2 for qarg1, qarg2 in zip(h_node.qargs, h_node.qargs[::-1])
    }

    for node in dag.topological_op_nodes():
        # if node == h_node:
        if DAGNode.semantic_eq(node, h_node):
            # TODO update to use _get_node_cns

            if node.name == "cx":
                new_dag.apply_operation_back(cx_replace.to_instruction(), node.qargs)
            elif node.name == "iswap":
                new_dag.apply_operation_back(iswap_replace.to_instruction(), node.qargs)
            else:
                raise ValueError(f"Unsupported operation, {node.name}")
            flip_flag = True

        else:
            if flip_flag:
                new_dag.apply_operation_back(
                    node.op, [swap_wires.get(qarg, qarg) for qarg in node.qargs]
                )
            else:
                new_dag.apply_operation_back(node.op, node.qargs)
    # fix with a swap
    # new_dag.apply_operation_back(SwapGate(), h_node.qargs)
    return new_dag
