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
    Unroll3qOrMore,
    VF2Layout,
)
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from transpile_benchy.passmanagers.abc_runner import CustomPassManager

from mirror_gates.cns_sabre_v3 import ParallelSabreSwapMS  # , SabreSwapMS
from mirror_gates.qiskit.sabre_layout import SabreLayout
from mirror_gates.sqiswap_equiv import sel  # noqa: F401
from mirror_gates.utilities import (
    AssignAllParameters,
    FastConsolidateBlocks,
    RemoveAllMeasurements,
    RemoveIGates,
    RemoveSwapGates,
    SaveCircuitProgress,
)

# 20,20 is what Qiskit uses for level 3
LAYOUT_TRIALS = 1  # (physical CPU_COUNT)
SWAP_TRIALS = 4
SEED = 7


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

    def build_pre_stage(self, **kwargs) -> PassManager:
        """Pre-process the circuit before running."""
        pm = PassManager()
        pm.property_set = kwargs.get("property_set", {})
        pm.append(RemoveBarriers())
        pm.append(AssignAllParameters())
        pm.append(Unroll3qOrMore())
        pm.append(OptimizeSwapBeforeMeasure())
        pm.append(RemoveIGates())
        pm.append(RemoveSwapGates())
        pm.append(RemoveDiagonalGatesBeforeMeasure())
        pm.append(RemoveFinalMeasurements())
        # We don't have classical bit registers implemented yet
        pm.append(RemoveAllMeasurements())  # hacky cleaning
        pm.append(SaveCircuitProgress("pre"))
        # NOTE, could consolidate in main stage .requires
        # but if we do it here we won't have to repeat for each restart loop
        pm.append(FastConsolidateBlocks(coord_caching=True))
        return pm

    def build_post_stage(self, **kwargs) -> PassManager:
        """Post-process the circuit after running."""
        pm = PassManager()
        pm.property_set = kwargs.get("property_set", {})
        pm.append(SaveCircuitProgress("post"))
        # consolidate before metric depth pass
        # NOTE this is required because of the QiskitRunner unrolling
        pm.append(FastConsolidateBlocks(coord_caching=True))
        return pm

    class QiskitRunner:
        """Run stock transpiler on the circuit.

        This is used to see if there exist any optimizations or cancellations.
        NOTE: Qiskit here won't know that the virtual-swaps are free,
        could be a problem - but we are forcing the input and those unitaries
        have already been coded into the DAGOpNodes - so it can't get rid of them.
        """

        @classmethod
        def _build_stage(cls, **kwargs):
            stage = cls()
            stage.property_set = kwargs.get("property_set", {})
            stage.coupling = kwargs.get("coupling_map", None)
            stage.basis_gates = kwargs.get("basis_gates", None)
            return stage

        def run(self, circuit):
            """Run the transpiler on the circuit."""
            return transpile(
                circuit,
                coupling_map=self.coupling,
                optimization_level=3,
                # basis_gates=self.basis_gates,
                basis_gates=["u", "cx", "swap", "id"],
                initial_layout=self.property_set.get("post_layout", None),
            )

    def _run_stage(self, stage_builder, circuit):
        """Run a stage and update the property set."""
        # FIXME, only QiskitRunner needs coupling_map and basis_gates
        # maybe a better way to move attributes around?
        stage = stage_builder(
            property_set=self.property_set,
            coupling_map=self.coupling,
            basis_gates=self.basis_gates,
        )
        circuit = stage.run(circuit)
        if stage.property_set:
            self.property_set.update(stage.property_set)
        return circuit

    def run(self, circuit):
        """Run the transpiler on the circuit.

        NOTE: the super class run method is overridden here to allow for
        the interruption between main- and post- processing to accommodate
        for Qiskit's optimization level 3 transpiler.
        """
        self.property_set = {}  # reset property set

        circuit = self._run_stage(self.build_pre_stage, circuit)
        circuit = self._run_stage(self.build_main_stage, circuit)
        circuit = self._run_stage(self.QiskitRunner._build_stage, circuit)
        circuit = self._run_stage(self.build_post_stage, circuit)
        circuit = self._run_stage(self.build_metric_stage, circuit)

        # accepted_subs missing if QiskitRunner is used
        # or if VF2Layout is called
        if "accepted_subs" not in self.property_set:
            self.property_set["accepted_subs"] = 0

        return circuit


class SabreMS(CustomLayoutRoutingManager):
    """SabreMS pass manager."""

    def __init__(self, coupling, parallel=True, cx_basis=False, logger=None):
        """Initialize the pass manager.

        Use parallel=False for debugging.
        """
        self.parallel = parallel
        self.name = "SABREMS"
        super().__init__(coupling, cx_basis=cx_basis, logger=logger)

    def build_main_stage(self, **kwargs):
        """Run SabreMS."""
        pm = PassManager()
        pm.property_set = kwargs.get("property_set", {})

        # Create the SabreMS pass
        routing_method = ParallelSabreSwapMS(
            coupling_map=self.coupling,
            trials=SWAP_TRIALS,
            basis_gate=self.basis_gate,
            parallel=self.parallel,
            seed=SEED,
        )

        # Create layout_method
        layout_method = SabreLayout(
            coupling_map=self.coupling,
            routing_pass=routing_method,
            layout_trials=LAYOUT_TRIALS,
            seed=SEED,
        )

        # VF2Layout
        pm.append(VF2Layout(coupling_map=self.coupling, seed=SEED, call_limit=int(3e7)))

        def vf2_not_converged(property_set):
            return (
                property_set["VF2Layout_stop_reason"]
                is not VF2LayoutStopReason.SOLUTION_FOUND
            )

        # Append the SabreMS pass with the condition
        pm.append(layout_method, condition=vf2_not_converged)
        pm.append(FullAncillaAllocation(self.coupling))
        pm.append(EnlargeWithAncilla())
        pm.append(ApplyLayout())
        pm.append(routing_method, condition=vf2_not_converged)
        pm.append(SaveCircuitProgress("mid"))
        return pm


class QiskitLevel3(CustomLayoutRoutingManager):
    """Qiskit level 3 pass manager."""

    def __init__(self, coupling, cx_basis=False):
        """Initialize the pass manager."""
        self.name = "Qiskit"
        super().__init__(coupling, cx_basis=cx_basis)

    def build_main_stage(self, **kwargs):
        """Do nothing.

        NOTE: just do nothing here,
        then the QiskitRunner will handle placement and routing.
        transpile() has initial_layout=None if post_layout has not been set.
        """
        return PassManager()
