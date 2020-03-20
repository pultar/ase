import numpy as np 
import matplotlib.pyplot as plt

def spectrum(atom,kind='ks',show=True,outfile='spectrum.png'):
    """
    This functions sketch a 'band' plot for
    a give kind of energies (ks,qp,singlet and triplet)
    It is possible to visualise it on the run (default True)
    It is possible to save the plot (put name in outfile)
    """
    if (kind == 'ks'):
        label = 'Kohn-Sham Energies'
        energies = atom.get_ks_energies()
    elif (kind == 'qp'):
        label = 'Quasiparticle Energies'
        energies = atom.get_qp_energies()
    elif (kind == 'singlet'):
        label = 'Singlet Energies'
        energies = atom.get_singlet_energies()
    elif (kind == 'triplet'):
        label = 'Triple Energies'
        energies = atom.get_triplet_energies()
    ax = plt.gca()
    ax.hlines(energies,-0.5,0.5)
    ax.set_xticks([])
    ax.set_xlabel(label)
    ax.set_ylabel('Energy [eV]')
    if (show==True):
        plt.show()
    plt.savefig(outfile)


def self_energy(atom,show=True,outfile='selfenergy.png'):
    """
    This function sketches the self-energy correction
    respect to the Kohn-Sham energies
    """
    ks = atom.get_ks_energies()
    qp = atom.get_qp_energies()
    ks = ks[:len(qp)]
    plt.scatter(ks,qp-ks)
    if (show==True):
        plt.show()
    plt.savefig(outfile)




