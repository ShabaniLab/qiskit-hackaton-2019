"""Circuit initialization routines.

"""
from math import pi
import numpy as np


def initialize_chain(circuit, qreg, zeeman, mode='logical_zero'):
    """Initialize the chain of qubit.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Circuit used to represent the evolution of the chain.
    qreg : qiskit.QuantumRegister
        Quantum register describing the qubits, the last qubit is always the
        coupler.
    initial_zeeman : np.ndarray
        Initial Zeeman field per site.
    mode : {'logical_zero', 'logical_one', 'up' 'down'}

    """
    N = len(zeeman)  # total size of spin chain
    ferromagnetic_sites = [int(i) for i in np.where(np.less(zeeman, 1))[0]]

    # Preparing the state
    if mode == 'logical_zero':
        circuit.h(qreg[ferromagnetic_sites[0]])
    elif mode == 'logical_one':
        circuit.ry(-pi/2, qreg[ferromagnetic_sites[0]])
    elif mode == 'down':
        for i in ferromagnetic_sites:
            circuit.x(qreg[i])

    if mode.startswith('logical'):
        first_ferro = ferromagnetic_sites[0]
        # Iterate on the position of the sites inside the chain (ie we put
        # the first site at zero and account for the offset later)
        for i in range(1, len(ferromagnetic_sites)):
            # Number of cnot in each iteration
            for k in range(np.int(np.exp2(i-1))):
                if (np.int(k+np.exp2(i-1)) >=
                    np.int64(len(ferromagnetic_sites))):
                    break
                # Real index of the control qubit in the chain.
                index = first_ferro + k
                circuit.cx(qreg[index], qreg[np.int(index + np.exp2(i-1))])

    for j in [i for i in range(N) if i not in ferromagnetic_sites]:
        circuit.h(qreg[j])
