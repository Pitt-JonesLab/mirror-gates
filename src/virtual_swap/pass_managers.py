"""Pre-defined pass managers for benchmarking."""

# from qiskit import transpile
# from qiskit.transpiler.passmanager import PassManager
from abc import ABC, abstractmethod

from qiskit import transpile
from qiskit.transpiler.basepasses import AnalysisPass
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
    SabreLayout,
    SabreSwap,
    Unroller,
)
from slam.utils.transpiler_pass.weyl_decompose import RootiSwapWeylDecomposition

# this code is buggy, see https://github.com/Qiskit/qiskit-terra/pull/9375
# I can't use this version bc qiskit version missing DAGCircuit functionality
from transpile_benchy.runner import AbstractRunner

from virtual_swap.cns_sabre_v2 import CNS_SabreSwap_V2

# from virtual_swap.deprecated.cns_brute import CNS_Brute
from virtual_swap.deprecated.sabre_swap import SabreSwap
from virtual_swap.sqiswap_equiv import RemoveIGates


class SaveCircuitProgress(AnalysisPass):
    """Used to save the state of the circuit Mid-way through the transpiler,
    for debugging."""

    def run(self, dag):
        # convert dag to circuit,
        # save into property_set
        from qiskit.converters import dag_to_circuit

        self.property_set["circuit_progress"] = dag_to_circuit(dag)
        return dag


class LayoutRouteSqiswap(AbstractRunner, ABC):
    """Subclass for AbstractRunner implementing pre- and post-processing."""

    def __init__(self, coupling, logger=None):
        """Initialize the runner."""
        self.coupling = coupling
        self.logger = logger
        super().__init__()

    def pre_process(self):
        """Pre-process the circuit before running."""
        self.pm.append(RemoveIGates())
        self.pm.append(Unroller(["u", "cx", "iswap", "swap"]))

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
                # debug, save current circuit to property_set
                SaveCircuitProgress(),
                Collect2qBlocks(),
                ConsolidateBlocks(force_consolidate=True),
                RootiSwapWeylDecomposition(),
                Optimize1qGates(basis=["u", "cx", "iswap", "swap"]),
                # does not help for sqiswap, but maybe I need to add
                # something inside of this function?
                # not sure that any rules would apply
                CommutativeCancellation(),
            ]
        )

    @abstractmethod
    def main_process(self):
        """Abstract method for main processing."""
        pass

    def run(self, circuit):
        """Run the transpiler on the circuit."""
        try:
            return super().run(circuit)
        except Exception as e:
            print(e)
            return None


class SabreCNSV2(LayoutRouteSqiswap):
    """Sabre CNS V2 pass manager."""

    def main_process(self):
        """Run SabreCNSV2."""
        routing = CNS_SabreSwap_V2(
            self.coupling, heuristic="decay", preserve_layout=True
        )
        self.pm.append(SabreLayout(self.coupling, routing_pass=routing))
        self.pm.append(
            [FullAncillaAllocation(self.coupling), EnlargeWithAncilla(), ApplyLayout()]
        )
        self.pm.append(routing)

    def run(self, circuit):
        """Run the transpiler on the circuit."""
        try:
            return super().run(circuit)
        finally:
            if self.logger is not None:
                self.logger.info(
                    f"Accepted CNS subs: {self.pm.property_set['accept_subs']}"
                )


class SabreQiskit(LayoutRouteSqiswap):
    """Sabre Qiskit pass manager."""

    def main_process(self):
        """Run SabreQiskit."""
        routing = SabreSwap(self.coupling, heuristic="decay")
        self.pm.append(SabreLayout(self.coupling, routing_pass=routing))
        self.pm.append(
            [FullAncillaAllocation(self.coupling), EnlargeWithAncilla(), ApplyLayout()]
        )
        self.pm.append(routing)


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
class QiskitLevel3(QiskitTranspileRunner):
    def run(self, circuit):
        transp = transpile(circuit, coupling_map=self.coupling, optimization_level=3)
        return self.pm.run(transp)


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
