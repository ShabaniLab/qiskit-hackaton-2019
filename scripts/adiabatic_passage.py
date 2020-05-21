"""Study the impact of the chosen time step over the fidelity.

"""

# --- Parameters -----------------------------------------------------------------------

NUMBER_OF_SITES = 6

FERRO_DOMAIN_SIZE = 3

H_FERRO = 0.01

H_PARA = 5

ZEEMAN_UPDATE_DELAY = 2

GAP_FRACTION = 0.0

MIN_INCREMENTS = [0.02, 0.05, 0.1, 0.2, 0.5]

COUPLER_STRENGTH = 1.0

COUPLER_DURATION = 2

SHOTS = 10000

REPETITIONS = 3

TROTTER_STEP = 0.1

RESULTS_PATH = "adiabatic-passage.csv"

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
        incr,
        ZEEMAN_UPDATE_DELAY,
        int(round(ZEEMAN_UPDATE_DELAY / TROTTER_STEP)),
        method=strategy,
    )
    add_measurement(qcirc, qreg, creg, [-2 - i for i in range(FERRO_DOMAIN_SIZE)])

    backend = Aer.get_backend("qasm_simulator")
    job = execute(
        qcirc, backend, shots=SHOTS, backend_options={"max_parallel_threads": 4}
    )

    backend = Aer.get_backend("qasm_simulator")
    vals = []
    for i in range(REPETITIONS):
        job = execute(
            qcirc, backend, shots=SHOTS, backend_options={"max_parallel_threads": 4}
        )

        result = job.result()
        print(result.get_counts())
        if coupler == "-z":
            val = (result.get_counts().get("1" * FERRO_DOMAIN_SIZE, 0.0) / SHOTS) * 100
        else:
            val = (result.get_counts().get("0" * FERRO_DOMAIN_SIZE, 0.0) / SHOTS) * 100
        vals.append(val)
    return np.average(vals), np.std(vals)


if not os.path.isfile(RESULTS_PATH) or not USE_RESULTS:

    results = np.zeros((8, len(TROTTER_STEPS)))
    for i, incr in enumerate(MIN_INCREMENTS):
        print(f"Computation for Δh = {incr}")
        for j, (coupler, strategy) in enumerate(
            zip(("+z", "-z", "+z", "-z"), ("both", "both", "single", "single"))
        ):
            print(f"Computation for coupler: {coupler}, strategy: {strategy}")
            val, std = estimate_fidelity(incr, coupler, strategy)
            results[j, i] = val
            results[j, i] = std

    df = pd.DataFrame(
        {
            "Field increments": np.array(MIN_INCREMENTS),
            "Coupler: Up, Strategy: both": results[0],
            "Coupler: Down, Strategy: both": results[2],
            "Coupler: Up, Strategy: single": results[4],
            "Coupler: Down, Strategy: single": results[6],
            "Coupler: Up, Strategy: both - std": results[1],
            "Coupler: Down, Strategy: both - std": results[3],
            "Coupler: Up, Strategy: single - std": results[5],
            "Coupler: Down, Strategy: single - std": results[7],
        }
    )
    if RESULTS_PATH:
        df.to_csv(RESULTS_PATH)
else:
    df = pd.read_csv(RESULTS_PATH)

plt.figure(constrained_layout=True)
for column in df.columns[1:]:
    if column == "Field increments" or column.endswith("std"):
        continue
    plt.errorbar(
        df["Field increments"],
        df[column],
        df[column + " - std"],
        marker="*",
        label=column,
    )
plt.xscale("log")
plt.xlabel("Field increment")
plt.ylabel("Fidelity")
plt.legend()
plt.show()
