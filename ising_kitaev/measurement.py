"""Routines to measure the chain state.

"""
import numpy as np


def rotate_to_measurement_basis(qc, q, ferromagnetic_qubits, basis='logical'):
    """Add a projective measurement along the specified basis for some qubits.

    """
    if basis == 'logical':
        # Measurement ( Implement U^† )
        n = len(ferromagnetic_qubits)
        first_ferro = min(ferromagnetic_qubits)

        # Iterate on the position of the sites inside the chain  in reverse
        # order(ie we put the first site at zero and account for the offset
        # later)
        for i in reversed(range(1, n)):
            # num of cnot in each iteration
            for k in range(np.int(np.exp2(i-1))):
                if np.int(k+np.exp2(i-1)) >= np.int(n):
                    break
                index = first_ferro + k
                qc.cx(q[index], q[np.int(index+np.exp2(i-1))])

        qc.h(q[ferromagnetic_qubits[0]])

    return qc, q


def add_measurement(qc, qreg, creg, ferromagnetic_qubits):
    """Add the measurement of the ferromagnetic domain.

    """
    qc.barrier()
    for i, j in enumerate(ferromagnetic_qubits):
        qc.measure(qreg[j], creg[i])
