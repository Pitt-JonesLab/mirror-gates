"""Noisy fidelity of a circuit."""
import numpy as np
from qiskit import Aer, transpile
from qiskit.circuit import Delay
from qiskit.quantum_info import state_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ASAPSchedule
from qiskit_aer import AerSimulator

# Import from Qiskit Aer noise module
from qiskit_aer.noise import NoiseModel, RelaxationNoisePass, thermal_relaxation_error

# 100 microsec (in nanoseconds)
T1 = 100e3
# 100 microsec
T2 = 100e3

# Instruction times (in nanoseconds)
time_u3 = 25
time_cx = 100
time_siswap = time_cx / 2.0
# divide by 2 again since
# each sqrt(iSwap) is compiled to an RXX and RYY
time_rxx = time_siswap / 2.0


def get_noisy_fidelity(qc, coupling_map):
    """Get noisy fidelity of a circuit.

    Args:
        qc (QuantumCircuit): circuit to run
        coupling_map (CouplingMap): coupling map of device

    Returns:
        fidelity (float): noisy fidelity of circuit
        duration (int): duration of circuit
        circ (QuantumCircuit): transpiled circuit
    """
    N = coupling_map.size()
    basis_gates = ["cx", "u", "u3", "rxx", "ryy", "id"]

    # Step 1. Convert into simulator basis gates
    simulator = Aer.get_backend("aer_simulator")
    circ = transpile(
        qc,
        simulator,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
    )

    # Step 2. Create Noise Model, with instruction durations
    # T1 and T2 values for all qubits
    N = coupling_map.size()
    T1s = [T1] * N
    T2s = [T2] * N

    # (inst, qubits, time)
    instruction_durations = []

    # Add errors to noise model
    noise_thermal = NoiseModel(basis_gates=basis_gates)
    for j in range(N):
        error_u3 = thermal_relaxation_error(T1s[j], T2s[j], time_u3)
        noise_thermal.add_quantum_error(error_u3, ["u1", "u2", "u3", "u"], [j])
        instruction_durations.append(("u", j, time_u3))
        instruction_durations.append(("u3", j, time_u3))

    for j, k in coupling_map:
        error_cx = thermal_relaxation_error(T1s[j], T2s[j], time_cx).tensor(
            thermal_relaxation_error(T1s[k], T1s[k], time_cx)
        )
        error_rxx = thermal_relaxation_error(T1s[j], T2s[j], time_rxx).tensor(
            thermal_relaxation_error(T1s[k], T1s[k], time_rxx)
        )
        noise_thermal.add_quantum_error(error_cx, "cx", [j, k])
        noise_thermal.add_quantum_error(error_rxx, ["rxx", "ryy"], [j, k])
        instruction_durations.append(("cx", (j, k), time_cx))
        instruction_durations.append(("rxx", (j, k), time_rxx))
        instruction_durations.append(("ryy", (j, k), time_rxx))

    # print(noise_thermal)  ####

    instruction_durations.append(("save_density_matrix", list(range(N)), 0.0))
    noisy_simulator = AerSimulator(noise_model=noise_thermal)

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
    # NOTE Relaxation pass takes T1 and T2 in seconds
    pm.append(
        RelaxationNoisePass(
            list(np.array(T1s) * 10e-9),
            list(np.array(T2s) * 10e-9),
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

    return fidelity, duration, circ
