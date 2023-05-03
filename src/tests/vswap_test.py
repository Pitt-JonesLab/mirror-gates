"""Test the VirtualSWAP transpiler pass."""

from virtual_swap.vswap_pass2 import VirtualSwap
from qiskit import QuantumCircuit
# from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Unroller, TrivialLayout, ApplyLayout

class TestVirtualSwapPass():
    """Test the VirtualSwap transpiler pass."""
    def test_toffoli(self):
        # build a toffoli
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)

        # build a 2x2 square coupling map
        coupling = CouplingMap.from_line(4)

        # run the pass
        pm = PassManager()
        # need some basic unroll and layout
        pm.append([Unroller(['u', 'cx']), TrivialLayout(coupling), ApplyLayout()])
        pm.append(VirtualSwap(coupling))
        new_circ = pm.run(qc)

        # check the output has depth 10
        assert new_circ.depth() == 10