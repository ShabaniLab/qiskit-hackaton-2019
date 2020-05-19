"""Study the impact of the chosen time step over the fidelity.

"""

# --- Parameters -----------------------------------------------------------------------

NUMBER_OF_SITES = 12

FERRO_DOMAIN_SIZE = 3

H_FERRO = 0.01

H_PARA = 3

COUPLER = 1.2

ZEEMAN_UPDATE_DELAY = 2

GAP_FRACTION = 0.2

MIN_INCREMENT = 0.05

COUPLER_STRENGTH = 1.2

COUPLER_DURATION = 2

SHOTS = 10

TROTTER_STEPS = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.4, 0.5, 1]

RESULTS_PATH = ""

USE_RESULTS = True

# --- Execution ------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from ising_kitaev import (
    initialize_chain,
    initialize_coupler,
    braid_chain,
    rotate_to_measurement_basis,
    add_measurement,
)
import matplotlib.pyplot as plt
from qiskit import Aer, execute


def estimate_fidelity(timestep, use_logical_state, coupler, should_braid):
    """Estimate the fidelity of an operation.

    Parameters
    ----------
    timestep : float
        Timestep to use in the Trotter expansion.

    use_logical_state : bool
        Use logical state 0 as input otherwise use ↑↑↑

    coupler : bool
        Is the coupler present or absent from the chain

    should_braid : bool
        When teh coupler is present should we perform a braiding operation.

    """
    initial_config = np.array([H_FERRO] * 3 + [H_PARA] * 9)
    qreg = QuantumRegister(13)
    creg = ClassicalRegister(3)
    qcirc = QuantumCircuit(qreg, creg)
    initialize_chain(
        qcirc, qreg, initial_config, "logical_zero" if use_logical_state else "up"
    )
    initialize_coupler(qcirc, qreg)
    braid_chain(
        qcirc,
        qreg,
        np.pi,
        40,
        initial_config,
        COUPLER_STRENGTH if coupler else None,
        GAP_FRACTION,
        MIN_INCREMENT,
        ZEEMAN_UPDATE_DELAY,
        COUPLER_DURATION,
        int(round(ZEEMAN_UPDATE_DELAY / timestep)),
        int(round(COUPLER_DURATION / timestep)),
        method="both",
    )
    if use_logical_state:
        rotate_to_measurement_basis(qcirc, qreg, creg, [0, 1, 2])
    add_measurement(qcirc, qreg, creg, [0, 1, 2])

    backend = Aer.get_backend("qasm_simulator")
    job = execute(qcirc, backend, shots=SHOTS)

    result = job.result()
    if should_braid:
        val = (result.get_counts().get("111", 0.0) / SHOTS) * 100
    else:
        val = (result.get_counts().get("000", 0.0) / SHOTS) * 100
    return val


if not os.path.isfile(RESULTS_PATH) or not USE_RESULTS:

    results = np.empty((5, len(TROTTER_STEPS)))
    for i, step in enumerate(TROTTER_STEPS):
        print(f"Computation for Δt = {step}")
        for j, (l, c, b) in enumerate(
            zip(
                (False, True, False, True, False),
                (False, False, True, True, True),
                (False, False, False, False, True),
            )
        ):
            print(f"Case: logical_state {l}, coupler {c}, braiding {b}")
            results[j, i] = estimate_fidelity(step, l, c, b)

    df = pd.DataFrame(
        {
            "Steps": np.array(TROTTER_STEPS),
            "No coupler - |↑↑↑>": results[0],
            "No coupler - |0>": results[1],
            "Coupler - |↑↑↑>": results[2],
            "Coupler - |0>": results[3],
            "Braiding": results[4],
        }
    )
    if RESULTS_PATH:
        df.to_csv(RESULTS_PATH)
else:
    df = pd.read_csv(RESULTS_PATH)

plt.figure(constrained_layout=True)
for column in df.columns[1:]:
    plt.plot(df["Steps"], df[column], marker="o", label=column)
plt.xlabel("Trotter step")
plt.ylabel("Fidelity")
plt.legend()
plt.show()
