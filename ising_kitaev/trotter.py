"""Routines used to implement the trotter evolution of the chain.

"""


def pair_interaction(qc, q, i0, i1, coup):
    """Gate representing the σ_z σ_z interaction.

    Since we normalize by J, the coupling is the trotter step.

    """
    qc.cx(q[i0], q[i1])
    qc.u1(coup, q[i1])
    qc.cx(q[i0], q[i1])


def zeeman_term(qc, q, i, mag):
    """Zeeman on site interaction along σ_x.

    """
    qc.rx(mag, q[i])


def coupler_hamiltonian(qc, q, i0, i1, interaction):
    """Implement the σ_z σ_z σ_x interaction with the coupler qubit.

    The coupler qubit is always the last coupling in the register.

    """
    coupler_index = len(q) - 1
    qc.cx(q[i1], q[i0])
    qc.h(q[coupler_index])
    qc.cx(q[i0], q[coupler_index])
    qc.rz(-interaction, q[coupler_index])
    qc.cx(q[i0], q[coupler_index])
    qc.h(q[coupler_index])
    qc.cx(q[i1], q[i0])


def interaction_hamiltonian(
    qc, q, coupling, zeeman, coupler_interaction=None, debug=False
):
    """Implement the interaction term on the chain.

    If a coupler_interaction is specified, the coupling between the two sites
    closest to the middle of the chain is implemented through the coupler qubit
    which by convention is always the last qubit of the register.

    """
    # NUmber of Ising sites in the chain
    n = len(zeeman)
    # Determine the index of the middle of the chain.
    m = int(len(q) / 2 - 1)

    # Build the list of function to call to add the interaction
    to_call = [
        (coupler_hamiltonian, (qc, q, j, j + 1, coupler_interaction))
        if coupler_interaction and j == m
        else (pair_interaction, (qc, q, j, j + 1, coupling))
        for j in range(0, n - 1)
    ]

    for f, args in to_call[::2]:
        f(*args)
    if debug:
        qc.barrier()
    for f, args in to_call[1::2]:
        f(*args)
    if debug:
        qc.barrier()


def zeeman_hamiltonian(qc, q, zeeman, debug=False):
    """Implement the Zeeman term on the chain.


    """
    n = len(zeeman)
    for j in range(n):
        zeeman_term(qc, q, j, zeeman[j])
    if debug:
        qc.barrier()


def trotter_step_1(qc, q, coupling, zeeman, coupler, debug=False):
    """Add a trotter step to the circuit.

    """
    zeeman_hamiltonian(qc, q, zeeman, debug)
    interaction_hamiltonian(qc, q, coupling, zeeman, coupler, debug)


def trotter_step_2(qc, q, coupling, zeeman, coupler, debug=False):
    """Add a trotter step to the circuit.

    """
    zeeman_hamiltonian(qc, q, 0.5 * zeeman, debug)

    interaction_hamiltonian(qc, q, coupling, zeeman, coupler, debug)

    zeeman_hamiltonian(qc, q, 0.5 * zeeman, debug)


def trotter_step_4(qc, q, coupling, zeeman, interaction, debug=False):
    """Add a trotter step to the circuit.

    """
    s2 = 1 / (4 - 4 ** (1 / 3))
    trotter_step_2(qc, q, s2 * coupling, s2 * zeeman, s2 * interaction, debug)
    trotter_step_2(qc, q, s2 * coupling, s2 * zeeman, s2 * interaction, debug)
    trotter_step_2(
        qc,
        q,
        (1 - 4 * s2) * coupling,
        (1 - 4 * s2) * zeeman,
        (1 - 4 * s2) * interaction,
        debug,
    )
    trotter_step_2(qc, q, s2 * coupling, s2 * zeeman, s2 * interaction, debug)
    trotter_step_2(qc, q, s2 * coupling, s2 * zeeman, s2 * interaction, debug)


def trotter_step_4_2(qc, q, coupling, zeeman, interaction, debug=False):
    """Add a trotter step to the circuit.

    Ths implementation can suffer from instabilities and is hence not used.

    """
    s2 = 1 / (2 - 2 ** (1 / 3))
    trotter_step_2(qc, q, coupling, s2 * zeeman, interaction, debug)
    trotter_step_2(qc, q, coupling, (1 - 2 * s2) * zeeman, interaction, debug)
    trotter_step_2(qc, q, coupling, s2 * zeeman, interaction, debug)


def trotter(qc, q, zeeman, interaction, dt, nsteps, order=1, debug=False):
    """Perform a Trotter evolution for a given number of timesteps.

    """
    if order == 1:
        t_step = trotter_step_1
    elif order == 2:
        t_step = trotter_step_2
    elif order == 4:
        t_step = trotter_step_4
    else:
        raise ValueError(f"Invalid trotter order: {order}")

    for i in range(nsteps):
        t_step(
            qc,
            q,
            dt,
            zeeman * dt,
            interaction * dt if interaction else interaction,
            debug,
        )
