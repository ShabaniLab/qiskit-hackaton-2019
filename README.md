# Emulating Majorana fermions braiding on a 1D Ising chain

Team members:
- Matthieu Dartiailh
- Eric Song
- Akhil Francis
- *Javad Shabani*

We follow the proposal in https://arxiv.org/pdf/1703.08224.pdf to
emulate a Majorana braiding operation on an Ising chain with ferromagnetic
interaction and a transverse field. Following the mentionned paper, we will
refer to the area of small transverse field in the which the exchange
interaction dominate as being ferromagnetic and the area of large transverse
field as paramagnetic.

Usually braiding is impossible in 1D but here the use of an additional coupler
spin, inducing a three spins interactions, allows to circumvent that limitation.

Useful definitions
------------------

We define the following quantity on the ferromagnetic domain:

- logical zero |0> : (|↑↑↑> + |↓↓↓>)/√2
- logical one |1> : |0> : (|↑↑↑> - |↓↓↓>)/√2

We map the |0> state of the qubit to the spin state |↑>.

When measuring in the logical basis of the ferromagnetic domain: the logical 0
corresponds to |000> and the logical 1 to |001>

In the notebook we use the following terms:
- extending: operation consisting in adding one site to the ferromagnetic
  domain.
- retracting: operation consisting in removing one site from the ferromagnetic
  domain.
- moving: adding a site on one side and removing a site from the opposite site
  in a single adiabatic evolution of the Zeeman field.

Installation instruction
------------------------

Run python setup.py develop in the repo root folder to install the required
modules to run the IPython notebooks.
