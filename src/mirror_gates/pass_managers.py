"""Pre-defined pass managers for benchmarking."""

from abc import ABC

from qiskit.circuit.library import CXGate, iSwapGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ApplyLayout,
    Collect2qBlocks,
    ConsolidateBlocks,
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
from transpile_benchy.passmanagers.qiskit_baseline import QiskitStage

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
LAYOUT_TRIALS = 6  # (physical CPU_COUNT) #1,4,7 for debug
SWAP_TRIALS = 6
SEED = 42


class CustomLayoutRoutingManager(CustomPassManager, ABC):
    """Subclass for CustomPassManager implementing pre- and post-processing."""

    def __init__(self, coupling, cx_basis=False, logger=None, use_fast_settings=True):
        """Initialize the pass manager."""
        super().__init__(name=self.name)

        # set up transpiler kwargs
        self.use_fast_settings = use_fast_settings
        self.coupling = coupling
        self.cx_basis = cx_basis
        self.logger = logger
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

    def stage_builder(self):
        """Build stages in a defined sequence."""

        def _builder():
            yield self.build_pre_stage()
            yield self.build_main_stage()
            yield QiskitStage.from_predefined_config(
                optimization_level=3,
                coupling_map=self.coupling,
                basis_gates=self.basis_gates,
                initial_layout=self.property_set.get("post_layout", None),
            )
            yield self.build_post_stage()

        return _builder

    # XXX
    # FIXME, kwargs property_set is either deprecated or broken

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
        pm.append(RemoveAllMeasurements())  # hacky cleaning # FIXME
        # pm.append(SaveCircuitProgress("pre"))

        # NOTE, could consolidate in main stage .requires
        # but if we do it here we won't have to repeat for each restart loop
        if self.use_fast_settings:
            pm.append(FastConsolidateBlocks(coord_caching=True))
        else:
            pm.append(Collect2qBlocks())
            pm.append(ConsolidateBlocks(force_consolidate=True))
        return pm

    def build_post_stage(self, **kwargs) -> PassManager:
        """Post-process the circuit after running."""
        pm = PassManager()
        pm.append(SaveCircuitProgress("post0"))
        pm.property_set = kwargs.get("property_set", {})
        # pm.append(SaveCircuitProgress("post"))

        # consolidate before metric depth pass
        # NOTE this is required because QiskitRunner will unroll to CX basis
        if self.use_fast_settings:
            pm.append(FastConsolidateBlocks(coord_caching=True))
        else:
            pm.append(Collect2qBlocks())
            pm.append(ConsolidateBlocks(force_consolidate=True))
        return pm

    def run(self, circuit):
        """Run the transpiler on the circuit."""
        circuit = super().run(circuit)

        # FIXME: either benchmarker uses default value
        # or we configure SubsMetric differently
        # accepted_subs missing if QiskitRunner is used or if VF2Layout succeeds
        if "accepted_subs" not in self.property_set:
            self.property_set["accepted_subs"] = 0

        return circuit


class SabreMS(CustomLayoutRoutingManager):
    """SabreMS pass manager."""

    def __init__(
        self,
        coupling,
        name=None,
        parallel=True,
        cx_basis=False,
        logger=None,
        use_fast_settings=True,
        cost_function="depth",
        anneal_routing=False,
    ):
        """Initialize the pass manager.

        Use parallel=False for debugging.
        """
        self.parallel = parallel
        self.name = name or "SABREMS"
        self.cost_function = cost_function
        self.anneal_routing = anneal_routing
        super().__init__(
            coupling,
            cx_basis=cx_basis,
            logger=logger,
            use_fast_settings=use_fast_settings,
        )

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
            use_fast_settings=self.use_fast_settings,
            cost_function=self.cost_function,
        )

        # Create layout_method
        layout_method = SabreLayout(
            coupling_map=self.coupling,
            routing_pass=routing_method,
            layout_trials=LAYOUT_TRIALS,
            seed=SEED,
            anneal_routing=self.anneal_routing,
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

    def stage_builder(self):
        """Build stages in a defined sequence."""

        def _builder():
            yield QiskitStage.from_predefined_config(
                optimization_level=3,
                coupling_map=self.coupling,
                basis_gates=self.basis_gates,
                initial_layout=self.property_set.get("post_layout", None),
            )
            yield self.build_post_stage()

        return _builder
