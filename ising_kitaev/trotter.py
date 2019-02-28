"""Routines used to implement the trotter evolution of the chain.

"""


def pair_interaction(qc, q, i0, i1, coup):
    """Gate representing the σ_z σ_z interaction.

    Since we normalize by J, the coupling is the trotter step.

    """
    qc.cx(q[i0],q[i1])
    qc.u1(coup,q[i1])
    qc.cx(q[i0],q[i1])


def zeeman_term(qc, q, i, mag):
    """Zeeman on site interaction along σ_x.

    """
    qc.rx(mag, q[i])


def chain_hamiltonian(qc, q, coupling, zeeman, debug=False):
    """Implement the chain Hamiltonian.

    """
    n = len(zeeman)
    for j in range(0, n, 2):
        pair_interaction(qc, q, j, j+1, coupling)
    for j in range(1, n-1, 2):
        pair_interaction(qc, q, j, j+1, coupling)
        if(debug):
            qc.barrier()
    for j in range(n):
        zeeman_term(qc, q, j, zeeman[j])
    if(debug):
        qc.barrier()


def interaction_hamiltonian(qc, q, i0, i1, interaction):
    """Implement the σ_z σ_z σ_x interaction with the coupler qubit.

    The coupler qubit is always the last coupling in the register.

    """
    coupler_index = len(q) - 1
    qc.cx(q[i1],q[i0])
    qc.h(q[coupler_index])
    qc.cx(q[i0],q[coupler_index])
    qc.rz(-interaction, q[coupler_index])
    qc.cx(q[i0],q[coupler_index])
    qc.h(q[coupler_index])
    qc.cx(q[i1],q[i0])


def trotter_step(qc, q, coupling, zeeman, interaction):
    """Add a trotter step to the circuit.

    """
    chain_hamiltonian(qc, q, coupling, zeeman)
    # Determine the index of the middle of the chain.
    m = int(len(q)/2-1)
    if (interaction!=0.0):
        interaction_hamiltonian(qc, q, m, m+1,interaction)


def trotter(qc, q, zeeman, interaction, dt, nsteps, debug=False):
    """Perform a Trotter evolution for a given number of timesteps.

    """
    for i in range(nsteps):
        trotter_step(qc, q, dt, zeeman*dt, interaction*dt)

