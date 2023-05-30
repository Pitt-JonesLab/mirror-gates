"""Pre-defined pass managers for benchmarking."""

# from qiskit import transpile
# from qiskit.transpiler.passmanager import PassManager
from abc import ABC, abstractmethod

from qiskit.transpiler.passes import (
    ApplyLayout,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    OptimizeSwapBeforeMeasure,
    SabreLayout,
    Unroller,
)

# this code is buggy, see https://github.com/Qiskit/qiskit-terra/pull/9375
# I can't use this version bc qiskit version missing DAGCircuit functionality
from transpile_benchy.runner import AbstractRunner

from virtual_swap.cns_sabre_v2 import CNS_SabreSwap_V2

# from virtual_swap.deprecated.cns_brute import CNS_Brute
from virtual_swap.deprecated.sabre_swap import SabreSwap
from virtual_swap.sqiswap_equiv import RemoveIGates


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
                OptimizeSwapBeforeMeasure(),
                # Collect2qBlocks(),
                # ConsolidateBlocks(force_consolidate=True),
                # RootiSwapWeylDecomposition(),
                # Optimize1qGates(basis=["u", "cx", "iswap", "swap"]),
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
        routing = CNS_SabreSwap_V2(self.coupling, heuristic="lookahead")
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
