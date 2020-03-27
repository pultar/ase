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

def GWspectrum(atom,show=True,outfile='GWspectrum.png'):
    """
    This functions sketch a 'band' plot for
    GW spectrum (KS vs. QP_pert vs. QP)
    It is possible to visualise it on the run (default True)
    It is possible to save the plot (put name in outfile)
    """
    plt.close("all")
    hrt2ev = 27.2214
    ks = hrt2ev*atom.get_ks_energies()
    qp_pert = hrt2ev*atom.get_qp_pert_energies()
    qp = hrt2ev*atom.get_qp_energies()
    plt.hlines(ks[:len(qp)],0,1)
    plt.hlines(qp_pert,1.5,2.5)
    plt.hlines(qp,3,4)
    plt.xticks([0.5,2,3.5],('KS','PQP','DQP'))
    plt.ylabel('Energy [eV]')
    if (show==True):
        plt.show()
    plt.savefig(outfile)
def self_energy(atom,show=True,outfile='selfenergy.png'):
    """
    This function sketches the self-energy correction
    respect to the Kohn-Sham energies
    """
    plt.close("all")
    ks = atom.get_ks_energies()
    qp = atom.get_qp_energies()
    ks = ks[:len(qp)]
    plt.scatter(ks,qp-ks,marker = 'o', alpha=0.7, c = 'red')
    plt.ylabel('QP - KS (eV)')
    plt.xlabel('KS (eV)')
    if (show==True):
        plt.show()
    plt.savefig(outfile)

def gaussian(x, center, fwhm):
    """ Computes the value of a gaussian with center x and fwhm at position x. """
    # FWHM = 2*sqrt(2 ln2) sigma = 2.3548 sigma
    sigma = fwhm / 2.3548
    return(np.exp(-0.5 * ((x - center) / sigma)**2) / sigma / np.sqrt(2.0 * np.pi))
def lorentzian(x, center, fwhm):
    """ Computes the value of a lorentzian with center x and fwhm at position x. """
    return(fwhm *fwhm / ((x - center)**2 + fwhm * fwhm) / np.pi)

def epsilon_2(atom,omega_in=0.1,omega_fin=30,num=300,fwhm=1e-2,axis = 0,outfile='BSEepsilon.png',show=True):
    hrt2ev = 27.2214
    singlet = hrt2ev*atom.get_singlet_energies()
    tdipoles = atom.get_transition_dipoles()[:,axis]
    w = np.linspace(omega_in,omega_fin,num=num)
    eps2 = []
    for x in w:
        c = 0
        for a,b in zip(singlet,tdipoles):
            c += lorentzian(x,a,fwhm)*b**2
        eps2.append(c)
    plt.close("all")
    plt.plot(w,eps2)
    if (show==True):
        plt.show()
    plt.savefig(outfile)
