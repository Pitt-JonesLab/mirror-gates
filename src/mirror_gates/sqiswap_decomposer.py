"""Decompose 2Q gates into SiSwap gates.

Relies on this branch: https://github.com/evmckinney9/qiskit-evmckinney9/tree/sqisw-gate
where I merged this PR (https://github.com/Qiskit/qiskit/pull/9375)
into qiskit-terra 0.24.2
"""

from qiskit.converters import circuit_to_dag
from qiskit.synthesis.su4 import SiSwapDecomposer
from qiskit.transpiler.basepasses import TransformationPass

from mirror_gates.fast_unitary import FastConsolidateBlocks

decomp = SiSwapDecomposer(euler_basis=["u"])


class SiSwapDecomposePass(TransformationPass):
    """Decompose 2Q gates into SiSwap gates."""

    def __init__(self):
        """Initialize the SiSwapDecomposePass pass."""
        super().__init__()
        self.requires = [FastConsolidateBlocks(coord_caching=True)]

    def run(self, dag):
        """Run the SiSwapDecomposePass pass on `dag`."""
        # for every 2Q gate
        for node in dag.two_qubit_ops():
            decomp_node = decomp(node.op)
            decomp_dag = circuit_to_dag(decomp_node)
            # dag.substitute_node(node, decomp_node)
            dag.substitute_node_with_dag(node, decomp_dag)

        return dag
