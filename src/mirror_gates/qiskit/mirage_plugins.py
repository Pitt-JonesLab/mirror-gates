"""Legacy plugin for SabreSwap."""
from typing import Optional
from mirror_gates.mirage import ParallelMirage
from mirror_gates.qiskit.sabre_swap import SabreSwap as LegacySabreSwap
from mirror_gates.sabre_layout_v2 import SabreLayout
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler import PassManager, PassManagerConfig
from mirror_gates.utilities import (
    AssignAllParameters,
    RemoveAllMeasurements,
    RemoveIGates,
    RemoveSwapGates,
)
from qiskit.transpiler.passes import (
    OptimizeSwapBeforeMeasure,
    RemoveBarriers,
    RemoveDiagonalGatesBeforeMeasure,
    RemoveFinalMeasurements,
    Unroll3qOrMore,
)
from mirror_gates.fast_unitary import FastConsolidateBlocks

class LegacySabreLayoutPlugin(PassManagerStagePlugin):
    """Version of Python implementation of SabreLayout that has parallel layout trials
    
    Default behvaior is to have no layout trials when given a custom routing method.
    We use this so that LegacySabre can run with multiple layout trials.
    """
    def pass_manager(self, pass_manager_config: PassManagerConfig, 
                     optimization_level: int = None) -> PassManager:
        """Return the layout stage pass manager."""
        layout_pm = PassManager()
        # use basic heuristic for routing
        routing = ParallelMirage(pass_manager_config.coupling_map, heuristic="basic")
        # use legacy SabreSwap for layout
        routing.atomic_routing = LegacySabreSwap
        layout_pm.append(SabreLayout(pass_manager_config.coupling_map, routing_pass=routing))
        layout_pm += common.generate_embed_passmanager(pass_manager_config.coupling_map)
        return layout_pm

# NOTE we would have liked to consolidate routing into layout
# this is how is done in pass_managers.py, but here we need skip_routing=True
# I found separate routing and layout stages was required when using plugins this way

class MirageRoutingPlugin(PassManagerStagePlugin):
    """Qiskit plugin for Mirage transpiler routing."""
    def pass_manager(self, pass_manager_config: PassManagerConfig, 
                     optimization_level: int = None) -> PassManager:
        routing_pm = PassManager()
        routing = ParallelMirage(pass_manager_config.coupling_map, trials=20)
        routing_pm += common.generate_routing_passmanager(routing,
                                                          pass_manager_config.target,
                                                          coupling_map=pass_manager_config.coupling_map,
                                                          check_trivial=True,)
    
class MirageLayoutPlugin(PassManagerStagePlugin):
    """Qiskit plugin for Mirage transpiler layout."""
    def pass_manager(self, pass_manager_config: PassManagerConfig, 
                     optimization_level: int = None) -> PassManager:
        """Return the layout stage pass manager."""
        layout_pm = PassManager()

        # some setup required for mirage
        # NOTE not all these may be necessary, 
        # but keeping same from pass_managers.py for now
        layout_pm.append(RemoveBarriers())
        layout_pm.append(AssignAllParameters())
        layout_pm.append(Unroll3qOrMore())
        layout_pm.append(OptimizeSwapBeforeMeasure())
        layout_pm.append(RemoveIGates())
        layout_pm.append(RemoveSwapGates())
        layout_pm.append(RemoveDiagonalGatesBeforeMeasure())
        layout_pm.append(RemoveFinalMeasurements())
        layout_pm.append(RemoveAllMeasurements())
        layout_pm.append(FastConsolidateBlocks(coord_caching=True))

        routing = ParallelMirage(pass_manager_config.coupling_map, trials=20)
        layout_pm.append(SabreLayout(pass_manager_config.coupling_map, 
                                     routing_pass=routing, 
                                     skip_routing=True, 
                                     layout_trials=20))
        layout_pm += common.generate_embed_passmanager(pass_manager_config.coupling_map)
        return layout_pm