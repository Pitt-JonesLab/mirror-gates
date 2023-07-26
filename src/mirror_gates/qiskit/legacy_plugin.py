"""Legacy plugin for SabreSwap."""
from typing import Optional
from mirror_gates.qiskit.sabre_swap import SabreSwap as LegacySabreSwap
from mirror_gates.cns_sabre_v3 import ParallelSabreSwapMS
from mirror_gates.qiskit.sabre_layout_v2 import SabreLayout as LegacySabreLayout
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler import PassManager, PassManagerConfig

class LegacySabrePlugin(PassManagerStagePlugin):
    """Python version of SabreSwap as a transpiler plugin.
    
    In order to use the python version of SabreSwap (for timing considerations),
    we need to define the LegacySabreSwap pass as a qiskit plugin.
    """
    def pass_manager(self, pass_manager_config: PassManagerConfig,
                     optimization_level: int = None) -> PassManager:
        """Return the routing stage pass manager."""
        # somewhat hacky use of atomic_routing attribute
        # we do this because LegacySabreSwap does not have trials parameter
        routing = ParallelSabreSwapMS(pass_manager_config.coupling_map)
        routing.atomic_routing = LegacySabreSwap
        return common.generate_routing_passmanager(
                routing,
                pass_manager_config.target,
                coupling_map=pass_manager_config.coupling_map,
            )

class LegacySabreLayoutPlugin(PassManagerStagePlugin):
    """Version of SabreLayout that has parallel layout trials
    
    Default behvaior is to have no layout trials when given a custom routing method.
    We use this so that LegacySabre can run with 20 layout trials.
    """
    def pass_manager(self, pass_manager_config: PassManagerConfig, 
                     optimization_level: int = None) -> PassManager:
        """Return the layout stage pass manager."""
        layout_pm = PassManager()
        routing = ParallelSabreSwapMS(pass_manager_config.coupling_map, heuristic="basic")
        routing.atomic_routing = LegacySabreSwap
        layout_pm.append(LegacySabreLayout(pass_manager_config.coupling_map, routing_pass=routing))
        layout_pm += common.generate_embed_passmanager(pass_manager_config.coupling_map)
        return layout_pm