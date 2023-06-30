"""Pre-defined pass managers for benchmarking."""

from abc import ABC

from qiskit import transpile
from qiskit.circuit.library import CXGate, iSwapGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    OptimizeSwapBeforeMeasure,
    RemoveBarriers,
    RemoveDiagonalGatesBeforeMeasure,
    RemoveFinalMeasurements,
    RemoveResetInZeroState,
    Unroller,
)
from transpile_benchy.passmanagers.abc_runner import CustomPassManager

from mirror_gates.cns_sabre_v3 import ParallelSabreSwapMS  # , SabreSwapMS
from mirror_gates.qiskit.sabre_layout import SabreLayout
from mirror_gates.sqiswap_equiv import sel  # noqa: F401
from mirror_gates.utilities import AssignAllParameters, RemoveIGates

LAYOUT_TRIALS = 6  # (physical CPU_COUNT)
SWAP_TRIALS = 6


class CustomLayoutRoutingManager(CustomPassManager, ABC):
    """Subclass for CustomPassManager implementing pre- and post-processing."""

    def __init__(self, coupling, cx_basis=False, logger=None):
        """Initialize the pass manager."""
        self.coupling = coupling
        self.logger = logger
        self.cx_basis = cx_basis
        if self.cx_basis:
            self.basis_gate = CXGate()
            self.gate_costs = 1.0
            self.name += r"-$\texttt{CNOT}$"
            self.basis_gates = ["u", "cx", "id"]
        else:
            self.basis_gate = iSwapGate().power(1 / 2)
            self.gate_costs = 0.5
            self.name += r"-$\sqrt{\texttt{iSWAP}}$"
            self.basis_gates = ["u", "xx_plus_yy", "id"]
        super().__init__(name=self.name)

    def build_pre_stage(self) -> PassManager:
        """Pre-process the circuit before running."""
        pm = PassManager()
        pm.append(RemoveIGates())
        pm.append(RemoveBarriers())
        pm.append(RemoveFinalMeasurements())
        pm.append(AssignAllParameters())
        pm.append(Unroller(["u", "u3", "cx", "iswap", "swap"]))
        pm.append(OptimizeSwapBeforeMeasure())
        pm.append(RemoveResetInZeroState())
        pm.append(RemoveDiagonalGatesBeforeMeasure())
        return pm

    def build_post_stage(self) -> PassManager:
        """Post-process the circuit after running."""
        pm = PassManager()
        # need to unroll for consolidate blocks to work
        pm.append(Unroller(["u", "cx", "iswap", "swap"]))

        # I don't think these are necessary
        # after we already have Qiskit's optimization level 3
        # pm.append(Unroller(["u", "cx", "iswap", "swap"]))
        # pm.append(CommutativeCancellation())
        # pm.append(RemoveResetInZeroState())
        # pm.append(OptimizeSwapBeforeMeasure())
        # pm.append(RemoveDiagonalGatesBeforeMeasure())
        # pm.append(Optimize1qGates(basis=["u", "cx", "iswap", "swap"]))
        # pm.append(SaveCircuitProgress())
        # pm.append(Collect2qBlocks())
        # pm.append(ConsolidateBlocks(force_consolidate=True))
        return pm

    class QiskitRunner:
        """Run stock transpiler on the circuit."""

        def __init__(self, coupling, basis_gates):
            """Initialize the runner."""
            self.coupling = coupling
            self.basis_gates = basis_gates
            self.property_set = {}

        def run(self, circuit):
            """Run the transpiler on the circuit."""
            return transpile(
                circuit,
                coupling_map=self.coupling,
                optimization_level=3,
                basis_gates=self.basis_gates,
                initial_layout=self.property_set.get("post_layout", None),
            )

    def run(self, circuit):
        """Run the transpiler on the circuit.

        NOTE: the super class run method is overridden here to allow for
        the interruption between main- and post- processing to accommodate
        for Qiskit's optimization level 3 transpiler.
        """
        self.property_set = {}  # reset property set
        stages = [
            self.build_pre_stage(),
            self.build_main_stage(),
            self.QiskitRunner(self.coupling, self.basis_gates),
            self.build_post_stage(),
            self.build_metric_stage(),
        ]
        for stage in stages:
            stage.property_set = self.property_set
            circuit = stage.run(circuit)
            self.property_set.update(stage.property_set)

        # patch cleanup
        if "Qiskit" in self.name:
            self.property_set["accepted_subs"] = 0

        return circuit


class SabreMS(CustomLayoutRoutingManager):
    """SabreMS pass manager."""

    def __init__(self, coupling, parallel=True, cx_basis=False, logger=None):
        """Initialize the pass manager."""
        self.parallel = parallel
        self.name = "SABREMS"
        super().__init__(coupling, cx_basis=cx_basis, logger=logger)

    def build_main_stage(self):
        """Run SabreMS."""
        pm = PassManager()

        # # single-shot
        # routing_method = SabreSwapMS(coupling_map=self.coupling)

        routing_method = ParallelSabreSwapMS(
            coupling_map=self.coupling,
            trials=SWAP_TRIALS,
            basis_gate=self.basis_gate,
            parallel=self.parallel,
        )

        layout_method = SabreLayout(
            coupling_map=self.coupling,
            routing_pass=routing_method,
            layout_trials=LAYOUT_TRIALS,
        )
        pm.append(layout_method)
        pm.append(FullAncillaAllocation(self.coupling))
        pm.append(EnlargeWithAncilla())
        pm.append(ApplyLayout())
        pm.append(routing_method)
        return pm


class QiskitLevel3(CustomLayoutRoutingManager):
    """Qiskit level 3 pass manager."""

    def __init__(self, coupling, cx_basis=False):
        """Initialize the pass manager."""
        self.name = "Qiskit"
        super().__init__(coupling, cx_basis=cx_basis)

    def build_main_stage(self):
        """Do nothing.

        NOTE: just do nothing here,
        then the QiskitRunner will handle placement and routing.
        transpile() has initial_layout=None if post_layout has not been set.
        """
        return PassManager()
