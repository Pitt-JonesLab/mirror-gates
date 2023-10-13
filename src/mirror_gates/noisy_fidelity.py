"""Noisy fidelity of a circuit."""
import numpy as np
from qiskit import transpile
from qiskit.circuit import Delay
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import state_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ASAPSchedule, Optimize1qGatesDecomposition
from qiskit_aer import AerSimulator, QasmSimulator

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (
    NoiseModel,
    RelaxationNoisePass,
    depolarizing_error,
    thermal_relaxation_error,
)

from mirror_gates.logging import transpile_benchy_logger
from mirror_gates.sqiswap_decomposer import SiSwapDecomposePass

# See Github Issue: ...
# 80 microsec (in nanoseconds)
T1 = 80e3
# 80 microsec
T2 = 80e3

# Instruction times (in nanoseconds)
time_u3 = 25
time_cx = 100
time_siswap = int(time_cx / 2.0)
# divide by 2 again since
# each sqrt(iSwap) is compiled to an RXX and RYY
time_rxx = int(time_siswap / 2.0)

p1 = 0.0
p2 = 0.00658


class NoiseModelBuilder:
    """A class to help build a custom NoiseModel from scratch.

    Many of the functions are based on examples from
    https://github.com/Qiskit/qiskit-presentations/blob/master/2019-02-26_QiskitCamp/QiskitCamp_Simulation.ipynb
    """

    def __init__(self, basis_gates, coupling_map=None):
        """Initialize a NoiseModelBuilder."""
        self.noise_model = NoiseModel(basis_gates=basis_gates)
        self.coupling_map = coupling_map

    def construct_basic_device_model(self, p_depol1, p_depol2, t1, t2):
        """Emulate qiskit.providers.aer.noise.device.models.basic_device_noise_model().

        The noise model includes the following errors:

            * Single qubit readout errors on measurements.
            * Single-qubit gate errors consisting of a depolarizing error
              followed by a thermal relaxation error for the qubit the gate
              acts on.
            * Two-qubit gate errors consisting of a 2-qubit depolarizing
              error followed by single qubit thermal relaxation errors for
              all qubits participating in the gate.

        :param p_depol1: Probability of a depolarising error on single qubit gates
        :param p_depol2: Probability of a depolarising error on two qubit gates
        :param t1: Thermal relaxation time constant
        :param t2: Dephasing time constant
        """
        if t2 > 2 * t1:
            raise ValueError("t2 cannot be greater than 2t1")

        # Thermal relaxation error

        # QuantumError objects
        error_thermal_u3 = thermal_relaxation_error(t1, t2, time_u3)
        error_thermal_cx = thermal_relaxation_error(t1, t2, time_cx).expand(
            thermal_relaxation_error(t1, t2, time_cx)
        )
        error_thermal_rxx = thermal_relaxation_error(t1, t2, time_rxx).expand(
            thermal_relaxation_error(t1, t2, time_rxx)
        )

        # Depolarizing error
        error_depol1 = depolarizing_error(p_depol1, 1)
        error_depol2 = depolarizing_error(p_depol2, 2)

        self.noise_model.add_all_qubit_quantum_error(
            error_depol1.compose(error_thermal_u3), "u"
        )

        for pair in self.coupling_map:
            self.noise_model.add_quantum_error(
                error_depol2.compose(error_thermal_cx), "cx", pair
            )
            self.noise_model.add_quantum_error(
                error_depol2.compose(error_thermal_rxx), ["rxx", "ryy"], pair
            )


def heuristic_fidelity(N, duration):
    """Get heuristic fidelity of a circuit."""
    decay_factor = (1 / T1 + 1 / T2) * duration
    single_qubit_fidelity = np.exp(-decay_factor)
    total_fidelity = single_qubit_fidelity**N
    return total_fidelity


def get_noisy_fidelity(qc, coupling_map, sqrt_iswap_basis=False):
    """Get noisy fidelity of a circuit.

    NOTE: if qc is too big, will use heuristic fidelity function.

    Args:
        qc (QuantumCircuit): circuit to run, assumes all gates are consolidated
        coupling_map (CouplingMap): coupling map of device

    Returns:
        fidelity (float): noisy fidelity of circuit
        duration (int): duration of circuit
        circ (QuantumCircuit): transpiled circuit
        expected_fidelity (float): expected fidelity of circuit
    """
    N = coupling_map.size()
    num_active = len(list(circuit_to_dag(qc).idle_wires())) - qc.num_clbits
    basis_gates = ["cx", "u", "rxx", "ryy", "id"]

    # Step 0. Create Noise Model

    # 0A. Set up Instruction Durations
    # (inst, qubits, time)
    instruction_durations = []
    for j in range(N):
        instruction_durations.append(("u", j, time_u3))
    for j, k in coupling_map:
        instruction_durations.append(("cx", (j, k), time_cx))
        instruction_durations.append(("rxx", (j, k), time_rxx))
        instruction_durations.append(("ryy", (j, k), time_rxx))
    instruction_durations.append(("save_density_matrix", list(range(N)), 0.0))

    # 0B. If circuit is too big, use heuristic fidelity function
    # Use heuristic fidelity function
    circ = transpile(
        qc,
        basis_gates=basis_gates,
        instruction_durations=instruction_durations,
        scheduling_method="asap",
        coupling_map=coupling_map,
    )
    duration = circ.duration
    expected_fidelity = heuristic_fidelity(num_active, duration)
    if N > 10:
        return 0, duration, circ, expected_fidelity
    else:
        transpile_benchy_logger.debug(f"Expected fidelity: {expected_fidelity:.4g}")

    # 0C. Build noise model
    builder = NoiseModelBuilder(basis_gates, coupling_map)
    builder.construct_basic_device_model(p_depol1=p1, p_depol2=p2, t1=T1, t2=T2)
    noise_model = builder.noise_model

    # 0D. Create noisy simulator
    noisy_simulator = AerSimulator(noise_model=noise_model)

    # Step 1. Given consolidated circuit, decompose into basis gates
    if sqrt_iswap_basis:
        decomposer = PassManager()
        decomposer.append(SiSwapDecomposePass())
        decomposer.append(Optimize1qGatesDecomposition())
        qc = decomposer.run(qc)

    # Step 2. Convert into simulator basis gates
    # simulator = Aer.get_backend("density_matrix_gpu")
    simulator = QasmSimulator(method="density_matrix")
    circ = transpile(
        qc,
        simulator,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
    )

    # Step 3. transpile with scheduling and durations
    circ = transpile(
        qc,
        noisy_simulator,
        basis_gates=basis_gates + ["save_density_matrix"],
        instruction_durations=instruction_durations,
        scheduling_method="asap",
        coupling_map=coupling_map,
    )

    # Step 4. Relaxation noise for idle qubits
    pm = PassManager()
    pm.append(ASAPSchedule())
    pm.append(
        RelaxationNoisePass(
            t1s=[T1] * N,
            t2s=[T2] * N,
            dt=1e-9,
            op_types=[Delay],
        )
    )
    circ = pm.run(circ)
    duration = circ.duration

    # Step 5. Run perfect and noisy simulation and compare
    circ.save_density_matrix(list(range(N)))

    perfect_result = simulator.run(circ).result().data()["density_matrix"]
    noisy_result = noisy_simulator.run(circ).result().data()["density_matrix"]
    fidelity = state_fidelity(perfect_result, noisy_result)

    return fidelity, duration, circ, expected_fidelity
