"""Study the impact of the chosen time step over the fidelity.

"""

# --- Parameters -----------------------------------------------------------------------

NUMBER_OF_SITES = 6

FERRO_DOMAIN_SIZE = 3

H_FERRO = 0.01

H_PARA = 4

ZEEMAN_UPDATE_DELAY = 2

GAP_FRACTION = 0.0

MIN_INCREMENT = 0.05

COUPLER_STRENGTH = 1.0

COUPLER_DURATION = 2

SHOTS = 10000
# 0.01, 0.02, 0.04, 0.05,
TROTTER_STEPS = [0.02, 0.05, 0.1, 0.2, 0.5]
TROTTER_STEPS.reverse()

RESULTS_PATH = "test.csv"

USE_RESULTS = False

# --- Execution ------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from ising_kitaev import (
    initialize_chain,
    initialize_coupler,
    move_chain,
    rotate_to_measurement_basis,
    add_measurement,
)
import matplotlib.pyplot as plt
from qiskit import Aer, execute


def estimate_fidelity(timestep, coupler, strategy):
    """Estimate the fidelity of an operation.

    Parameters
    ----------
    timestep : float
        Timestep to use in the Trotter expansion.

    coupler : {"up", "down"}
        Is the coupler up or down

    strategy : {"both", "single"}
        Field update strategy.

    """
    initial_config = np.array(
        [H_FERRO] * FERRO_DOMAIN_SIZE + [H_PARA] * (NUMBER_OF_SITES - FERRO_DOMAIN_SIZE)
    )
    qreg = QuantumRegister(NUMBER_OF_SITES + 1)
    creg = ClassicalRegister(FERRO_DOMAIN_SIZE)
    qcirc = QuantumCircuit(qreg, creg)
    initialize_chain(qcirc, qreg, initial_config, "up")
    initialize_coupler(qcirc, qreg, coupler)
    move_chain(
        qcirc,
        qreg,
        initial_config,
        initial_config[::-1],
        COUPLER_STRENGTH,
        GAP_FRACTION,
        MIN_INCREMENT,
        ZEEMAN_UPDATE_DELAY,
        int(round(ZEEMAN_UPDATE_DELAY / timestep)),
        method=strategy,
    )
    add_measurement(qcirc, qreg, creg, [-2 - i for i in range(FERRO_DOMAIN_SIZE)])

    backend = Aer.get_backend("qasm_simulator")
    job = execute(
        qcirc, backend, shots=SHOTS, backend_options={"max_parallel_threads": 4}
    )

    result = job.result()
    print(result.get_counts())
    if coupler == "-z":
        val = (result.get_counts().get("1" * FERRO_DOMAIN_SIZE, 0.0) / SHOTS) * 100
    else:
        val = (result.get_counts().get("0" * FERRO_DOMAIN_SIZE, 0.0) / SHOTS) * 100
    return val


if not os.path.isfile(RESULTS_PATH) or not USE_RESULTS:

    results = np.zeros((4, len(TROTTER_STEPS)))
    for i, step in enumerate(TROTTER_STEPS):
        print(f"Computation for Δt = {step}")
        for j, (coupler, strategy) in enumerate(
            zip(("+z", "-z", "+z", "-z"), ("both", "both", "single", "single"))
        ):
            print(f"Computation for coupler: {coupler}, strategy: {strategy}")
            results[j, i] = estimate_fidelity(step, coupler, strategy)

    df = pd.DataFrame(
        {
            "Steps": np.array(TROTTER_STEPS),
            "Coupler: Up, Strategy: both": results[0],
            "Coupler: Down, Strategy: both": results[1],
            "Coupler: Up, Strategy: single": results[2],
            "Coupler: Down, Strategy: single": results[3],
        }
    )
    if RESULTS_PATH:
        df.to_csv(RESULTS_PATH)
else:
    df = pd.read_csv(RESULTS_PATH)

plt.figure(constrained_layout=True)
for column in df.columns[1:]:
    if column == "Steps":
        continue
    plt.plot(df["Steps"], df[column], marker="o", label=column)
plt.xscale("log")
plt.xlabel("Trotter step")
plt.ylabel("Fidelity")
plt.legend()
plt.show()
