from typing import Iterable
from qiskit.circuit import Clbit, Qubit
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit import Instruction

class VswapDAGOpNode(DAGOpNode):
    def __init__(self, op, qargs: Iterable[Qubit] = ..., cargs: Iterable[Clbit] = ...):
        super().__init__(op, qargs, cargs)
        self.metadata = {}
    
    @classmethod
    def from_DAGOpNode(cls, dag_op_node: DAGOpNode):
        return cls(dag_op_node.op, dag_op_node.qargs, dag_op_node.cargs)
    
class VsInstruction(Instruction):
    def __init__(self, name, num_qubits, num_clbits, params, duration=None, unit="dt", label=None):
        super().__init__(name, num_qubits, num_clbits, params, duration, unit, label)
        self.metadata = {}
    @classmethod
    def from_Instruction(cls, instruction: Instruction):
        return cls(instruction.name, instruction.num_qubits, instruction.num_clbits, instruction.params, instruction.duration, instruction.unit, instruction.label)