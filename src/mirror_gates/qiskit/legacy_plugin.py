"""Legacy plugin for SabreSwap."""
from mirror_gates.qiskit.sabre_swap import SabreSwap as LegacySabreSwap
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
        return common.generate_routing_passmanager(
                LegacySabreSwap(pass_manager_config.coupling_map),
                pass_manager_config.target,
                coupling_map=pass_manager_config.coupling_map,
            )
