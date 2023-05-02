"""Test the VirtualSWAP transpiler pass."""

from src.virtual_swap.vswap_pass2 import VirtualSwap
from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap

class TestVirtualSwapPass(QiskitTestCase):
    """Test the VirtualSwap transpiler pass."""
    def toffoli(self):
        # build a toffoli
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)

        # build a 2x2 square coupling map
        coupling = CouplingMap.from_grid(2,2)

        # run the pass
        pass_ = VirtualSwap(coupling)
        new_circ = pass_.run(qc)

        # check the output has depth 10
        self.assertEqual(new_circ.depth(), 10)