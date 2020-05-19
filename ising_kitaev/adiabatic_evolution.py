"""Implement the adiabatic switching of the local zeeman fields.

"""
from itertools import zip_longest
from collections import defaultdict

import numpy as np

from .trotter import trotter
from .coupler import mid_braiding_manipulation


def estimate_gap(zeeman):
    """Estimate the gap based on the current zeeman configuration.

    The estimate is a naive semiclassical estimate based on the difference
    between ↑↑↑→→ and ↑↑→→→, whose energy difference is |J-h| where h is the
    field of the middle spin.

    Parameters
    ----------
    zeeman : np.ndarray
        Zeeman field per site from which to estimate the gap.

    """
    return np.abs(np.min(np.abs(1 - zeeman)))


def run_adiabatic_zeeman_change(
    circuit,
    qreg,
    initial_zeeman,
    final_zeeman,
    coupler_inter,
    gap_fraction,
    min_increment,
    delay,
    trotter_step_number,
    trotter_order=1,
):
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

    """
    # Determine which sites should be updated
    zi = np.copy(initial_zeeman)
    zeeman_diff = final_zeeman - initial_zeeman

    # check that we have a symmetric update in case more than one site change
    min_z = np.min(zeeman_diff)
    max_z = np.max(zeeman_diff)
    if min_z and max_z:
        assert -min_z == max_z, "Non-symmetric transformation of Zeeman field"

    zeeman_distance = np.max(np.abs(zeeman_diff))
    zeeman_update_sign = np.sign(zeeman_diff)

    dt = delay / trotter_step_number
    # First evolve the system.
    trotter(
        circuit,
        qreg,
        initial_zeeman,
        coupler_inter,
        dt,
        trotter_step_number,
        order=trotter_order,
    )

    # Evolve the system till we reach the final zeeman.
    i = 0
    while zeeman_distance > 0:
        gap = estimate_gap(zi)
        zeeman_step = min(max(gap_fraction * gap, min_increment), zeeman_distance)
        zi += zeeman_update_sign * zeeman_step
        trotter(
            circuit,
            qreg,
            zi,
            coupler_inter,
            dt,
            trotter_step_number,
            order=trotter_order,
        )

        zeeman_distance -= zeeman_step


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
            changes["added"][np.min(np.abs(if_sites - index))] = index
        # The site was removed, compute the distance to the final chain and
        # store the index as value
        elif site == -1:
            changes["removed"][np.min(np.abs(ff_sites - index))] = index

    # For each pair of modications either create a zeeman with both
    # or two different if the method is single.
    # For addition we go from smallest to largest distance, for deletion from
    # largest to smallest
    zeemans = []
    previous_zeeman = initial_zeeman
    for added, removed in zip_longest(
        sorted(changes["added"]), reversed(sorted(changes["removed"]))
    ):
        if added:
            z = np.copy(previous_zeeman)
            index = changes["added"][added]
            z[index] = final_zeeman[index]
            zeemans.append(z)
            previous_zeeman = z
        if removed:
            z = np.copy(previous_zeeman)
            index = changes["removed"][removed]
            z[index] = final_zeeman[index]
            zeemans.append(z)
            previous_zeeman = z

    if method == "both":
        zeemans = zeemans[1::2]

    return zeemans


def move_chain(
    circuit,
    qreg,
    initial_zeeman,
    final_zeeman,
    coupler_inter,
    gap_fraction,
    min_increment,
    delay,
    trotter_step_number,
    trotter_order=1,
    method="both",
):
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
    zeemans = determine_intermediate_zeemans(initial_zeeman, final_zeeman, method)

    i_zeeman = initial_zeeman
    for zeeman in zeemans:
        run_adiabatic_zeeman_change(
            circuit,
            qreg,
            i_zeeman,
            zeeman,
            coupler_inter,
            gap_fraction,
            min_increment,
            delay,
            trotter_step_number,
            trotter_order,
        )
        i_zeeman = zeeman


def braid_chain(
    circuit,
    qreg,
    theta,
    step_number,
    initial_zeeman,
    coupler_inter,
    gap_fraction,
    min_increment,
    delay,
    rot_time,
    trotter_step_number,
    trotter_order=1,
    method="both",
):
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
    step_number : int
        Number of step to use to perfrom the rotation.
    initial_zeeman : np.ndarray
        Initial Zeeman field per site.
    coupler_inter : float
        Strength of the interaction with the coupler.
    gap_fraction : float
        By what fraction of the estimated gap to update the zeeman field on the
        affected sites.
    min_increment : float
        Minimal increment of the Zeeman field to perform to avoid getting
        stuck.
    delay : float
        Time between two update of the Zeeman field.
    rot_time : float
        Total time to take to perform the rotation of the coupler.
    trotter_step_number : int
        Number of Trotter step to perform between two fields updates.
    rot_trotter_step_number : int
        Number of Trotter step to perform to complete the rotation of the coupler.
    method : {'both', 'single'}
        Should the chain movement occurs only from one side at a time (the
        chain is always elongated first) or from both ends.

    """
    final_zeeman = initial_zeeman[::-1]
    move_chain(
        circuit,
        qreg,
        initial_zeeman,
        final_zeeman,
        coupler_inter,
        gap_fraction,
        min_increment,
        delay,
        trotter_step_number,
        trotter_order,
        method,
    )
    mid_braiding_manipulation(
        circuit,
        qreg,
        theta,
        step_number,
        final_zeeman,
        coupler_inter,
        rot_time,
        rot_trotter_step_number,
        trotter_order,
    )
    move_chain(
        circuit,
        qreg,
        final_zeeman,
        initial_zeeman,
        coupler_inter,
        gap_fraction,
        min_increment,
        delay,
        trotter_step_number,
        trotter_order,
        method,
    )
