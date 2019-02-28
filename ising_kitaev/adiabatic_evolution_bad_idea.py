"""Implement the adiabatic switching of the local zeeman fields.

"""
from itertools import zip_longest
from collections import defaultdict

import numpy as np

from .trotter import trotter
from .coupler import mid_braiding_manipulation


def run_adiabatic_zeeman_change(circuit, qreg, initial_zeeman, final_zeeman,
                                coupler_inter, min_increment, adiabatic_steps,
                                delay, trotter_step_number):
    """Adiabatically evolve the system between two field configurations.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Circuit used to represent the evolution of the chain.
    qreg : qiskit.QuantumRegister
        Quantum register describing the qubits, the last qubit is always the
        coupler.
    initial_zeeman : np.ndarray
        Initial Zeeman field per site.
    final_zeeman : np.ndarray
        Final Zeeman field per site.
    coupler_inter : float
        Strength of the interaction between the coupler and the two sites of
        the chains it is connected to.
    min_increment : float
        Minimal increment of the Zeeman field to perform to avoid getting
        stuck.
    adiabatic_steps : int
        Number of steps to take both below and above 1.
    delay : float
        Time between two update of the Zeeman field.
    trotter_step_number : int
        Number of Trotter step to perform between two fields updates.

    """
    # Determine which sites should be updated
    zi = np.copy(initial_zeeman)
    zeeman_diff = final_zeeman - initial_zeeman

    # check that we have a symmetric update in case more than one site change
    min_diff_z = np.min(zeeman_diff)
    max_diff_z = np.max(zeeman_diff)
    if min_diff_z and max_diff_z:
        assert -min_diff_z == max_diff_z,\
            "Non-symmetric transformation of Zeeman field"

    # Create geometric progression both below and above 1
    min_z = min((np.min(initial_zeeman), np.min(final_zeeman)))
    max_z = min((np.max(initial_zeeman), np.max(final_zeeman)))
    small_zeeman_values = 1/np.geomspace(1/(1-min_increment/2), 1/min_z,
                                         adiabatic_steps)[::-1]
    large_zeeman_values = np.geomspace((1+min_increment/2), max_z,
                                       adiabatic_steps)
    zeeman_values =\
        np.hstack((small_zeeman_values, large_zeeman_values))
    # Identify the sites where the zeeman is increasing (use the computed
    # fields) and sites where it decreases (use the computed fields but in
    # reverse order)
    zeeman_incr = np.where(np.sign(zeeman_diff) == 1)[0]
    zeeman_decr = np.where(np.sign(zeeman_diff) == -1)[0]

    dt = delay / trotter_step_number
    # First evolve the system.
    trotter(circuit, qreg, initial_zeeman, coupler_inter, dt,
            trotter_step_number)

    # Evolve the system till we reach the final zeeman.
    for incr_z, decr_z in zip(zeeman_values[1:], zeeman_values[::-1]):
        zi[zeeman_incr] = incr_z
        zi[zeeman_decr] = decr_z
        trotter(circuit, qreg, zi, coupler_inter, dt, trotter_step_number)



def determine_intermediate_zeemans(initial_zeeman, final_zeeman, method):
    """Determine the intermediate zeeman configurations between two states.

    Parameters
    ----------
    initial_zeeman : np.ndarray
        Initial Zeeman field per site.
    final_zeeman : np.ndarray
        Final Zeeman field per site.
    method : {'both', 'single'}
        Should the chain movement occurs only from one side at a time (the
        chain is always elongated first) or from both ends.

    """
    initial_ferro = np.less(initial_zeeman, 1).astype(np.int)
    final_ferro = np.less(final_zeeman, 1).astype(np.int)
    changed_sites = final_ferro - initial_ferro
    if_sites = np.array([i for i, val in enumerate(initial_ferro) if val == 1])
    ff_sites = np.array([i for i, val in enumerate(final_ferro) if val == 1])

    changes = defaultdict(dict)
    for index, site in enumerate(changed_sites):
        # The site was added, compute the distance to the initial chain and
        # store the index as value
        if site == 1:
            changes['added'][np.min(np.abs(if_sites - index))] = index
        # The site was removed, compute the distance to the final chain and
        # store the index as value
        elif site == -1:
            changes['removed'][np.min(np.abs(ff_sites - index))] = index


    # For each pair of modications either create a zeeman with both
    # or two different if the method is single.
    # For addition we go from smallest to largest distance, for deletion from
    # largest to smallest
    zeemans = []
    previous_zeeman = initial_zeeman
    for added, removed in zip_longest(sorted(changes['added']),
                                      reversed(sorted(changes['removed']))):
        if added:
            z = np.copy(previous_zeeman)
            index = changes['added'][added]
            z[index] = final_zeeman[index]
            zeemans.append(z)
            previous_zeeman = z
        if removed:
            z = np.copy(previous_zeeman)
            index = changes['removed'][removed]
            z[index] = final_zeeman[index]
            zeemans.append(z)
            previous_zeeman = z

    if method == 'both':
        zeemans = zeemans[1::2]

    return zeemans


def move_chain(circuit, qreg, initial_zeeman, final_zeeman, coupler_inter,
               min_increment, adiabatic_steps, delay, trotter_step_number,
               method='both'):
    """Move the chain by one site step.

    The initial and final configurations are deduced from the zeeman fields.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Circuit used to represent the evolution of the chain.
    qreg : qiskit.QuantumRegister
        Quantum register describing the qubits, the last qubit is always the
        coupler.
    initial_zeeman : np.ndarray
        Initial Zeeman field per site.
    final_zeeman : np.ndarray
        Final Zeeman field per site.
    gap_fraction : float
        By what fraction of the estimated gap to update the zeeman field on the
        affected sites.
    min_increment : float
        Minimal increment of the Zeeman field to perform to avoid getting
        stuck.
    delay : float
        Time between two update of the Zeeman field.
    trotter_step_number : int
        Number of Trotter step to perform between two fields updates.
    method : {'both', 'single'}
        Should the chain movement occurs only from one side at a time (the
        chain is always elongated first) or from both ends.

    """
    zeemans = determine_intermediate_zeemans(initial_zeeman, final_zeeman,
                                             method)

    i_zeeman = initial_zeeman
    for zeeman in zeemans:
        run_adiabatic_zeeman_change(circuit, qreg, i_zeeman, zeeman,
                                    coupler_inter, min_increment,
                                    adiabatic_steps,
                                    delay, trotter_step_number)
        i_zeeman = zeeman


def braid_chain(circuit, qreg, theta, initial_zeeman, coupler_inter,
                min_increment, adiabatic_steps, delay, trotter_step_number,
                method='both'):
    """Perform a full braiding operation on a properly initialized system

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Circuit used to represent the evolution of the chain.
    qreg : qiskit.QuantumRegister
        Quantum register describing the qubits, the last qubit is always the
        coupler.
    theta : float
        Rotation to perform on the coupler qubit midway in the braiding.
    initial_zeeman : np.ndarray
        Initial Zeeman field per site.
    final_zeeman : np.ndarray
        Final Zeeman field per site.
    gap_fraction : float
        By what fraction of the estimated gap to update the zeeman field on the
        affected sites.
    min_increment : float
        Minimal increment of the Zeeman field to perform to avoid getting
        stuck.
    delay : float
        Time between two update of the Zeeman field.
    trotter_step_number : int
        Number of Trotter step to perform between two fields updates.
    method : {'both', 'single'}
        Should the chain movement occurs only from one side at a time (the
        chain is always elongated first) or from both ends.

    """
    final_zeeman = initial_zeeman[::-1]
    move_chain(circuit, qreg, initial_zeeman, final_zeeman, coupler_inter,
               min_increment, adiabatic_steps, delay, trotter_step_number,
               method)
    mid_braiding_manipulation(circuit, qreg, theta)
    move_chain(circuit, qreg, final_zeeman, initial_zeeman, coupler_inter,
               min_increment, adiabatic_steps, delay, trotter_step_number,
               method)
