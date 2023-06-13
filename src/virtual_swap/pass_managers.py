"""Pre-defined pass managers for benchmarking."""

# from qiskit import transpile
# from qiskit.transpiler.passmanager import PassManager
from abc import ABC, abstractmethod

import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from qiskit.transpiler.passes import (
    ApplyLayout,
    Collect2qBlocks,
    CommutativeCancellation,
    ConsolidateBlocks,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGates,
    OptimizeSwapBeforeMeasure,
    RemoveDiagonalGatesBeforeMeasure,
    RemoveResetInZeroState,
    Unroller,
)
from slam.utils.transpiler_pass.weyl_decompose import RootiSwapWeylDecomposition

# this code is buggy, see https://github.com/Qiskit/qiskit-terra/pull/9375
# I can't use this version bc qiskit version missing DAGCircuit functionality
from transpile_benchy.runner import AbstractRunner

from virtual_swap.cns_sabre_v3 import ParallelSabreSwapVS

# from virtual_swap.deprecated.cns_brute import CNS_Brute
# from virtual_swap.deprecated.sabre_swap import SabreSwap
from virtual_swap.sabre_layout import SabreLayout

# from qiskit.transpiler.passes import SabreLayout
from virtual_swap.sqiswap_equiv import RemoveIGates

LAYOUT_TRIALS = 6  # (physical CPU_COUNT)
SWAP_TRIALS = 6  # 6  # makes it so much slower :(


class SaveCircuitProgress(AnalysisPass):
    """Used to save the state of the circuit Mid-way through the transpiler,
    for debugging."""

    def __init__(self, qc_name=None):
        super().__init__()
        self.qc_name = qc_name or "circuit_progress"

    def run(self, dag):
        # convert dag to circuit,
        # save into property_set
        from qiskit.converters import dag_to_circuit

        self.property_set[self.qc_name] = dag_to_circuit(dag)
        return dag


class AssignAllParameters(TransformationPass):
    def __init__(self):
        super().__init__()

    def run(self, dag):
        # for every parameter, assign a random value [0, 2pi]
        # not sure I good way to do this, do messy in meantime
        qc = dag_to_circuit(dag)
        for param in qc.parameters:
            qc.assign_parameters({param: np.random.uniform(0, 2 * np.pi)}, inplace=True)
        return circuit_to_dag(qc)


class LayoutRouteSqiswap(AbstractRunner, ABC):
    """Subclass for AbstractRunner implementing pre- and post-processing."""

    def __init__(self, coupling, logger=None):
        """Initialize the runner."""
        self.coupling = coupling
        self.logger = logger
        super().__init__()

    # NOTE, these u and u3s is somewhat of a monkey fix :P
    # I don't know why but Optimize1qGates was breaking

    def pre_process(self):
        """Pre-process the circuit before running."""
        self.pm.append(RemoveIGates())
        self.pm.append(AssignAllParameters())
        self.pm.append(Unroller(["u", "u3", "cx", "iswap", "swap"]))
        self.pm.append(OptimizeSwapBeforeMeasure())
        self.pm.append(RemoveResetInZeroState())
        self.pm.append(RemoveDiagonalGatesBeforeMeasure())

    def post_process(self):
        """Post-process the circuit after running."""
        pass
        self.pm.append(
            [
                # adding this unroller fixes issue
                # consolidate block was not pushing together
                # the iswap_primes and 2Q blocks
                Unroller(["u", "cx", "iswap", "swap"]),
                RemoveResetInZeroState(),
                OptimizeSwapBeforeMeasure(),
                RemoveDiagonalGatesBeforeMeasure(),
                # this 1Q optimize is unnecessary, keeping it for cleaner mid circuits
                Optimize1qGates(basis=["u", "cx", "iswap", "swap"]),
                # debug, save current circuit to property_set
                SaveCircuitProgress(),
                Collect2qBlocks(),
                ConsolidateBlocks(force_consolidate=True),
            ]
        )
        if not self.cx_basis:
            self.pm.append(
                [
                    RootiSwapWeylDecomposition(),
                    Optimize1qGates(basis=["u", "u3", "cx", "iswap", "swap"]),
                ]
            )
        else:
            self.pm.append(Unroller(["u", "cx"]))
            # self.pm.append(TwoQubitBasisDecomposer(gate=CXGate(), euler_basis = 'u')
            # Optimize1qGates(basis=["cx", "iswap", "swap"]),
        # does not help for sqiswap, but maybe I need to add
        # something inside of this function?
        # not sure that any rules would apply
        self.pm.append(CommutativeCancellation())

    @abstractmethod
    def main_process(self):
        """Abstract method for main processing."""
        pass

    def run(self, circuit):
        """Run the transpiler on the circuit."""
        return self.pm.run(circuit)
        try:
            return super().run(circuit)
        except Exception as e:
            print(e)
            return None


class SabreVS(LayoutRouteSqiswap):
    """Sabre CNS V2 pass manager."""

    def __init__(self, coupling, cx_basis=False, logger=None):
        self.cx_basis = cx_basis
        super().__init__(coupling, logger)

    def main_process(self):
        """Run SabreVS."""
        routing_method = ParallelSabreSwapVS(
            coupling_map=self.coupling, trials=SWAP_TRIALS
        )
        # routing_method = SabreSwapVS(coupling_map=self.coupling)
        layout_method = SabreLayout(
            coupling_map=self.coupling,
            routing_pass=routing_method,
            layout_trials=LAYOUT_TRIALS,
        )
        self.pm.append(layout_method)

        self.pm.append(
            [FullAncillaAllocation(self.coupling), EnlargeWithAncilla(), ApplyLayout()]
        )
        self.pm.append(routing_method)
        # self.pm.append(SaveCircuitProgress("mid0"))

    def run(self, circuit):
        """Run the transpiler on the circuit."""
        # return super().run(circuit)
        try:
            return super().run(circuit)
        finally:
            if self.logger is not None and self.pm.property_set:
                self.logger.info(
                    f"Accepted CNS subs: {self.pm.property_set['accepted_subs']}"
                )


class SabreQiskit(LayoutRouteSqiswap):
    """Sabre Qiskit pass manager."""

    def __init__(self, coupling, cx_basis=False):
        self.cx_basis = cx_basis
        super().__init__(coupling)

    def main_process(self):
        """Run SabreQiskit."""
        # override trials,
        # this is because when we override the routing for CNS, it doesn't allow parallel trials
        # we set trials to 1 here in the qiskit baseline for normalization
        # however, each transpiler still runs best of N runs
        # routing_method = SabreSwap(coupling_map= self.coupling, trials=NUM_TRIALS)
        # not specifying routing_pass, so it will use the default SabreSwap with trials=CPU_COUNT
        layout_method = SabreLayout(
            coupling_map=self.coupling,
            layout_trials=LAYOUT_TRIALS,
            swap_trials=SWAP_TRIALS,
        )
        self.pm.append(layout_method)
        # # NOTE, I think SabreLayout already does this
        # # NVM, only if routing_pass is None
        # self.pm.append(
        #     [FullAncillaAllocation(self.coupling), EnlargeWithAncilla(), ApplyLayout()]
        # )
        # self.pm.append(routing_method)


class QiskitTranspileRunner(LayoutRouteSqiswap):
    """Used to noop the pre-, main-, post- passes."""

    def pre_process(self):
        pass

    def main_process(self):
        pass

    @abstractmethod
    def run(self):
        """Abstract method for overloaded run method."""
        pass


# same as SabreQiskit, but need to test if level=3 has additional optimizations
# SabreQiskit is just Layout/Routing, doesn't look for whatever other cancellations
# that might be in level=3
# class QiskitLevel3(QiskitTranspileRunner):
#     def run(self, circuit):
#         from qiskit.transpiler.passes.routing import SabreSwap
#         # override trials,
#         # this is because when we override the routing for CNS, it doesn't allow parallel trials
#         # we set trials to 1 here in the qiskit baseline for normalization
#         # however, each transpiler still runs best of N runs
#         routing_method = SabreSwap(coupling_map= self.coupling, trials=1)
#         layout_method = SabreLayout(coupling_map= self.coupling, layout_trials=1)
#         transp = transpile(circuit, coupling_map=self.coupling, optimization_level=3, routing_method=routing_method, layout_method=layout_method)
#         return self.pm.run(transp)


# class BruteCNS(LayoutRouteSqiswap):
#     def __init__(self, coupling):
#         pm = PassManager()
#         pm.append(Unroller(["u", "cx", "iswap", "swap"]))
#         pm.append(TrivialLayout(coupling))
#         pm.append(CNS_Brute(coupling))
#         # pm.append(Unroller(["u", "cx", "iswap", "swap"]))
#         super().__init__(pm)

# class Baseline(CustomPassManager):
#     def __init__(self, coupling):
#         self.coupling = coupling
#         # NOTE, for some reason the StagedPassManager, created by level_3_pass_manager
#         # I cannot append my own pass to it
#         # config = PassManagerConfig(coupling_map=coupling, basis_gates=["cx", "u3"])
#         # self.pm = level_3_pass_manager(config)
#         # super().__init__(pm)

#     # override the run function, see NOTE above
#     def run(self, qc):
#         intermediate = transpile(
#             qc,
#             coupling_map=self.coupling,
#             basis_gates=["cx", "u3"],
#             optimization_level=3,
#         )
#         temp_pm = PassManager()

#         temp_pm.append(
#             [
#                 Collect2qBlocks(),
#                 ConsolidateBlocks(force_consolidate=True),
#                 RootiSwapWeylDecomposition(),
#             ]
#         )
#         return temp_pm.run(intermediate)
