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

from mirror_gates.fast_unitary import FastConsolidateBlocks
from mirror_gates.mirage import ParallelMirage
from mirror_gates.sabre_layout_v2 import SabreLayout
from mirror_gates.sqiswap_equiv import sel  # noqa: F401
from mirror_gates.utilities import (
    AssignAllParameters,
    RemoveAllMeasurements,
    RemoveIGates,
    RemoveSwapGates,
    SaveCircuitProgress,
)

# 20,20 is what Qiskit uses for level 3
DEFAULT_LAYOUT_TRIALS = 20  # (physical CPU_COUNT) #1,4,7 for debug
DEFAULT_FB_ITERS = 4
DEFAULT_SWAP_TRIALS = 20
DEFAULT_SEED = 42


class CustomLayoutRoutingManager(CustomPassManager, ABC):
    """Subclass for CustomPassManager implementing pre- and post-processing."""

    def __init__(
        self,
        coupling,
        cx_basis=False,
        logger=None,
        use_fast_settings=True,
        layout_trials=None,
        fb_iters=None,
        swap_trials=None,
    ):
        """Initialize the pass manager."""
        super().__init__(name=self.name)

        self.layout_trials = layout_trials or DEFAULT_LAYOUT_TRIALS
        self.fb_iters = fb_iters or DEFAULT_FB_ITERS
        self.swap_trials = swap_trials or DEFAULT_SWAP_TRIALS
        self.seed = DEFAULT_SEED

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
        if "layout_trials" not in self.property_set:
            self.property_set["layout_trials"] = []
            self.property_set["layout_trials_std"] = 0

        return circuit


# TODO: refactor to use plugins?, will let combine main into Qiskit stages
# Qiskit stage will use the custom layout and routing methods as plugins


class Mirage(CustomLayoutRoutingManager):
    """Mirage pass manager."""

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
        fixed_aggression=None,
        layout_trials=None,
        fb_iters=None,
        swap_trials=None,
    ):
        """Initialize the pass manager.

        Use parallel=False for debugging.
        """
        self.parallel = parallel
        self.name = name or "Mirage"
        self.cost_function = cost_function
        self.anneal_routing = anneal_routing
        self.fixed_aggression = fixed_aggression
        super().__init__(
            coupling,
            cx_basis=cx_basis,
            logger=logger,
            use_fast_settings=use_fast_settings,
            layout_trials=layout_trials,
            fb_iters=fb_iters,
            swap_trials=swap_trials,
        )

    def build_main_stage(self, **kwargs):
        """Run Mirage."""
        pm = PassManager()
        pm.property_set = kwargs.get("property_set", {})

        # Create the Mirage pass
        routing_method = ParallelMirage(
            coupling_map=self.coupling,
            trials=self.swap_trials,
            basis_gate=self.basis_gate,
            parallel=self.parallel,
            seed=self.seed,
            use_fast_settings=self.use_fast_settings,
            cost_function=self.cost_function,
            fixed_aggression=self.fixed_aggression,
        )

        # Create layout_method
        layout_method = SabreLayout(
            coupling_map=self.coupling,
            routing_pass=routing_method,
            layout_trials=self.layout_trials,
            seed=self.seed,
            anneal_routing=self.anneal_routing,
            max_iterations=self.fb_iters,
            parallel=False,  # XXX turn off because of BrokenPipeError, not sure why yet
        )

        # VF2Layout
        pm.append(
            VF2Layout(coupling_map=self.coupling, seed=self.seed, call_limit=int(3e7))
        )

        def vf2_not_converged(property_set):
            return (
                property_set["VF2Layout_stop_reason"]
                is not VF2LayoutStopReason.SOLUTION_FOUND
            )

        # Append the Mirage pass with the condition
        pm.append(layout_method, condition=vf2_not_converged)
        pm.append(FullAncillaAllocation(self.coupling))
        pm.append(EnlargeWithAncilla())
        pm.append(ApplyLayout())
        pm.append(routing_method, condition=vf2_not_converged)
        pm.append(SaveCircuitProgress("mid"))
        return pm


class QiskitLevel3(CustomLayoutRoutingManager):
    """Qiskit level 3 pass manager."""

    def __init__(self, coupling, cx_basis=False, python_sabre=False):
        """Initialize the pass manager."""
        self.name = "Qiskit"
        self.python_sabre = python_sabre
        super().__init__(coupling, cx_basis=cx_basis)

    def stage_builder(self):
        """Build stages in a defined sequence."""

        def _builder():
            yield QiskitStage.from_predefined_config(
                optimization_level=3,
                coupling_map=self.coupling,
                basis_gates=self.basis_gates,
                routing_method="legacy_sabre" if self.python_sabre else None,
                layout_method="legacy_layout" if self.python_sabre else None,
            )
            yield self.build_post_stage()

        return _builder
