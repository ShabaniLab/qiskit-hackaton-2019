"""Study the impact of the chosen time step over the fidelity.

"""

# --- Parameters -----------------------------------------------------------------------

NUMBER_OF_SITES = 6

FERRO_DOMAIN_SIZE = 3

H_FERRO = 0.01

H_PARA = 5

ZEEMAN_UPDATE_DELAY = 2

GAP_FRACTION = 0.0

MIN_INCREMENT = 0.05

COUPLER_STRENGTH = 1.0

COUPLER_DURATION = 2

SHOTS = 10000

REPETITIONS = 10

TROTTER_STEPS = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.4, 0.5, 1, 2]
TROTTER_STEPS.reverse()

RESULTS_PATH = "trotter_step_evaluation.csv"

USE_RESULTS = False

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
        When the coupler is present should we perform a braiding operation.

    """
    initial_config = np.array(
        [H_FERRO] * FERRO_DOMAIN_SIZE + [H_PARA] * (NUMBER_OF_SITES - FERRO_DOMAIN_SIZE)
    )
    qreg = QuantumRegister(NUMBER_OF_SITES + (1 if coupler else 0))
    creg = ClassicalRegister(FERRO_DOMAIN_SIZE)
    qcirc = QuantumCircuit(qreg, creg)
    initialize_chain(
        qcirc, qreg, initial_config, "logical_zero" if use_logical_state else "up"
    )
    if coupler:
        initialize_coupler(qcirc, qreg)
    braid_chain(
        qcirc,
        qreg,
        np.pi,
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
        rotate_to_measurement_basis(qcirc, qreg, list(range(FERRO_DOMAIN_SIZE)))
    add_measurement(qcirc, qreg, creg, list(range(FERRO_DOMAIN_SIZE)))

    backend = Aer.get_backend("qasm_simulator")
    vals = []
    for i in range(REPETITIONS):
        job = execute(
            qcirc, backend, shots=SHOTS, backend_options={"max_parallel_threads": 4}
        )

        result = job.result()
        print(result.get_counts())
        if should_braid:
            val = (result.get_counts().get("1" * FERRO_DOMAIN_SIZE, 0.0) / SHOTS) * 100
        else:
            val = (result.get_counts().get("0" * FERRO_DOMAIN_SIZE, 0.0) / SHOTS) * 100
        vals.append(val)
    return np.average(vals), np.std(vals)


if not os.path.isfile(RESULTS_PATH) or not USE_RESULTS:

    results = np.zeros((10, len(TROTTER_STEPS)))
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
            val, std = estimate_fidelity(step, l, c, b)
            results[2 * j, i] = val
            results[2 * j + 1, i] = std

    df = pd.DataFrame(
        {
            "Steps": np.array(TROTTER_STEPS),
            "No coupler - |↑↑↑>": results[0],
            "No coupler - |0>": results[2],
            "Coupler - |↑↑↑>": results[4],
            "Coupler - |0>": results[6],
            "Braiding": results[8],
            "No coupler - |↑↑↑> - std": results[1],
            "No coupler - |0> - std": results[3],
            "Coupler - |↑↑↑> - std": results[5],
            "Coupler - |0> - std": results[7],
            "Braiding - std": results[9],
        }
    )
    if RESULTS_PATH:
        df.to_csv(RESULTS_PATH)
else:
    df = pd.read_csv(RESULTS_PATH)

plt.figure(constrained_layout=True)
for column in df.columns[1:]:
    if column == "Steps" or column.endswith("std"):
        continue
    plt.errorbar(
        df["Steps"], df[column], df[column + " - std"], marker="*", label=column
    )
plt.xscale("log")
plt.xlabel("Trotter step")
plt.ylabel("Fidelity")
plt.legend()
plt.show()
