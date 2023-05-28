"""Pre-defined pass managers for benchmarking."""

from qiskit import transpile
from qiskit.transpiler.passes import (
    ApplyLayout,
    Collect2qBlocks,
    ConsolidateBlocks,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGates,
    OptimizeSwapBeforeMeasure,
    SabreLayout,
    TrivialLayout,
    Unroller,
)
from qiskit.transpiler.passmanager import PassManager

# this code is buggy, see https://github.com/Qiskit/qiskit-terra/pull/9375
# I can't use this version bc qiskit version missing DAGCircuit functionality
from slam.utils.transpiler_pass.weyl_decompose import RootiSwapWeylDecomposition

from virtual_swap.deprecated.cns_brute import CNS_Brute
from virtual_swap.deprecated.sabre_swap import SabreSwap
from virtual_swap.passes.cns_sabre_v2 import CNS_SabreSwap_V2


class CustomPassManager(PassManager):
    def __init__(self, pm):
        self.pm = pm

        # force this ending, such that output is normalized
        # we want all benchy PMs to count sqiswap gates
        self.pm.append(
            [
                OptimizeSwapBeforeMeasure(),
                Optimize1qGates(basis=["u", "cx", "iswap", "swap"]),
                # Collect2qBlocks(),
                # ConsolidateBlocks(force_consolidate=True),
                # RootiSwapWeylDecomposition(),
            ]
        )

    def run(self, qc):
        return self.pm.run(qc)


class BruteCNS(CustomPassManager):
    def __init__(self, coupling):
        pm = PassManager()
        pm.append(Unroller(["u", "cx", "iswap", "swap"]))
        pm.append(TrivialLayout(coupling))
        pm.append(CNS_Brute(coupling))
        # pm.append(Unroller(["u", "cx", "iswap", "swap"]))
        super().__init__(pm)


class SabreCNSV2(CustomPassManager):
    def __init__(self, coupling):
        pm = PassManager()
        # XXX, force SABRE to only have 1 1Q gate between 2Q gates
        # temp fix :)
        pm.append(Optimize1qGates(["u", "cx", "iswap", "swap"]))
        routing = CNS_SabreSwap_V2(coupling, heuristic="lookahead")
        pm.append(SabreLayout(coupling, routing_pass=routing))
        pm.append(
            [FullAncillaAllocation(coupling), EnlargeWithAncilla(), ApplyLayout()]
        )
        pm.append(routing)
        pm.append(Unroller(["u", "cx", "iswap", "swap"]))
        super().__init__(pm)

    def run(self, qc):
        transp_qc = super().run(qc)
        print("Accepted CNS subs", self.pm.property_set["accept_subs"])
        return transp_qc


class SabreQiskit(CustomPassManager):
    def __init__(self, coupling):
        pm = PassManager()
        pm.append(Unroller(["u", "cx", "iswap", "swap"]))
        routing = SabreSwap(coupling, heuristic="decay")
        pm.append(SabreLayout(coupling, routing_pass=routing))
        pm.append(
            [FullAncillaAllocation(coupling), EnlargeWithAncilla(), ApplyLayout()]
        )
        pm.append(routing)
        pm.append(Unroller(["u", "cx", "iswap", "swap"]))
        super().__init__(pm)


class Baseline(CustomPassManager):
    def __init__(self, coupling):
        self.coupling = coupling
        # NOTE, for some reason the StagedPassManager, created by level_3_pass_manager
        # I cannot append my own pass to it
        # config = PassManagerConfig(coupling_map=coupling, basis_gates=["cx", "u3"])
        # self.pm = level_3_pass_manager(config)
        # super().__init__(pm)

    # override the run function, see NOTE above
    def run(self, qc):
        intermediate = transpile(
            qc,
            coupling_map=self.coupling,
            basis_gates=["cx", "u3"],
            optimization_level=3,
        )
        temp_pm = PassManager()

        temp_pm.append(
            [
                Collect2qBlocks(),
                ConsolidateBlocks(force_consolidate=True),
                RootiSwapWeylDecomposition(),
            ]
        )
        return temp_pm.run(intermediate)
