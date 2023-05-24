"""Pre-defined pass managers for benchmarking."""

from qiskit import transpile
from qiskit.transpiler.passes import (
    ApplyLayout,
    Collect2qBlocks,
    ConsolidateBlocks,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    SabreLayout,
    Unroller,
    OptimizeSwapBeforeMeasure,
)
from qiskit.transpiler.passmanager import PassManager

# this code is buggy, see https://github.com/Qiskit/qiskit-terra/pull/9375
# I can't use this version bc qiskit version missing DAGCircuit functionality
from slam.utils.transpiler_pass.weyl_decompose import RootiSwapWeylDecomposition

from virtual_swap.cns_sabre import CNS_SabreSwap
from virtual_swap.deprecated.sabre_swap import SabreSwap


class CustomPassManager(PassManager):
    def __init__(self, pm):
        self.pm = pm

        # force this ending, such that output is normalized
        # we want all benchy PMs to count sqiswap gates
        # self.pm.append(
        #     [
        #         Collect2qBlocks(),
        #         ConsolidateBlocks(force_consolidate=True),
        #         RootiSwapWeylDecomposition(),
        #     ]
        # )

    def run(self, qc):
        return self.pm.run(qc)


class SabreCNS(CustomPassManager):
    def __init__(self, coupling):
        pm = PassManager()
        routing = CNS_SabreSwap(coupling, heuristic="decay")
        pm.append(SabreLayout(coupling, routing_pass=routing))
        pm.append(
            [FullAncillaAllocation(coupling), EnlargeWithAncilla(), ApplyLayout()]
        )
        pm.append(routing)
        pm.append(OptimizeSwapBeforeMeasure())
        pm.append(Unroller(["u", "cx", "iswap", "swap"]))
        super().__init__(pm)

    def run(self, qc):
        transp_qc = super().run(qc)
        print("Accepted CNS subs", self.pm.property_set["accept_subs"])
        return transp_qc


class SabreQiskit(CustomPassManager):
    def __init__(self, coupling):
        pm = PassManager()
        routing = SabreSwap(coupling, heuristic="decay")
        pm.append(SabreLayout(coupling, routing_pass=routing))
        pm.append(
            [FullAncillaAllocation(coupling), EnlargeWithAncilla(), ApplyLayout()]
        )
        pm.append(routing)
        pm.append(OptimizeSwapBeforeMeasure())
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
