"""Coupler manipulation related routines.

The coupler is always assumed to be the last qubit in the register.

"""
from math import pi

from .trotter import trotter

def initialize_coupler(circuit, qreg):
    """Initialize the coupler such that e need only one forward-backward move.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Circuit used to represent the evolution of the chain.
    qreg : qiskit.QuantumRegister
        Quantum register describing the qubits, the last qubit is always the
        coupler.

    """
    circuit.rx(pi/2, qreg[len(qreg)-1])


def mid_braiding_manipulation(circuit, qreg, theta, step_number, zeeman,
                              coupler_inter, delay, trotter_step_number):
    """Manipulation performed on the coupler at mid-braiding.

    circuit : qiskit.QuantumCircuit
        Circuit used to represent the evolution of the chain.
    qreg : qiskit.QuantumRegister
        Quantum register describing the qubits, the last qubit is always the
        coupler.
    theta : float
        Rotation to perform on the coupler qubit midway in the braiding.
    step_number : int
        Number of step to use to perfrom the rotation.
    zeeman : np.ndarray
        Zeeman field per site.
    coupler_inter : float
        Strength of the interaction with the coupler.
    delay : float
        Time between two update of the Zeeman field.
    trotter_step_number : int
        Number of Trotter step to perform between two fields updates.

    """
    dt = delay / trotter_step_number
    for i in range(step_number):
        trotter(circuit, qreg, zeeman, coupler_inter, dt,
                trotter_step_number)
        circuit.ry(theta/step_number, qreg[len(qreg)-1])
