def test_slab_rneb():
    from ase.build import fcc100, add_adsorbate
    from ase.constraints import FixAtoms
    from ase.calculators.emt import EMT
    from ase.neb import NEB
    from ase.optimize import BFGS
    import numpy as np

    from ase.rneb import RNEB

    rneb = RNEB(logfile=None)

    # 3x3-Al(001) surface with 3 layers and an
    # Au atom adsorbed in a hollow site:
    slab = fcc100('Al', size=(3, 3, 3))
    slab.center(axis=2, vacuum=4.0)

    # Fix second and third layers:
    mask = [atom.tag > 1 for atom in slab]
    constraint = FixAtoms(mask=mask)
    slab.set_constraint(constraint)

    add_adsorbate(slab, 'Au', 1.7, 'hollow')
    add_adsorbate(slab, 'Au', 1.7, 'hollow', offset=(1, 0))
    initial_unrelaxed = slab.copy()
    initial_unrelaxed.pop(-1)

    # Use EMT potential:
    slab.set_calculator(EMT())

    # Initial state:
    initial_relaxed = initial_unrelaxed.copy()
    initial_relaxed.set_calculator(EMT())
    qn = BFGS(initial_relaxed, logfile=None)
    qn.run(fmax=0.05)

    # Final state:
    final_unrelaxed = slab.copy()
    final_unrelaxed.pop(-2)

    # RNEB symmetry identification
    sym_ops = rneb.find_symmetries(slab, initial_unrelaxed, final_unrelaxed)
    # check that path has reflection symmetry for each images
    images = create_path(initial_unrelaxed, final_unrelaxed)
    neb = NEB(images)
    neb.interpolate(images)
    sym_ops = rneb.reflect_path(images, sym=sym_ops)
    print(sym_ops)

    # Obtain the final relaxed structure with RNEB
    final_relaxed = rneb.get_final_image(slab, initial_unrelaxed,
                                         initial_relaxed, final_unrelaxed)

    images = create_path(initial_relaxed, final_relaxed)

    neb = NEB(images, sym=True, rotations=sym_ops[0])
    neb.interpolate()
    qn = BFGS(neb, logfile=None)
    qn.run(fmax=0.05)

    sym_image_energy = images[-2].get_potential_energy()

    # Do a normal NEB
    neb_images = create_path(initial_relaxed, final_relaxed)
    normal_neb = NEB(neb_images)
    normal_neb.interpolate()
    qn = BFGS(normal_neb, logfile=None)
    qn.run(fmax=0.05)

    normal_image_energy = neb_images[-2].get_potential_energy()

    assert np.isclose(sym_image_energy, normal_image_energy)


def create_path(init, final):
    from ase.calculators.emt import EMT

    images = [init]
    for i in range(3):
        image = init.copy()
        image.set_calculator(EMT())
        images.append(image)

    images.append(final)
    return images

def test_is_reflective():
    """ Test to check paths separately for reflection symmetry """
    from ase.build import fcc111
    
    atoms = fcc111('Cu', [2,2,1], 4, periodic=True)
    atoms = atoms.repeat([2,2,1])
    
    # give path to look at as path=(init_indice, final_indice)
    get_num_sym_operators(atoms, path=(1, 2)) # here i get 4
    get_num_sym_operators(atoms, path=(0, 1)) # here i get only 2!?

def get_num_sym_operators(atoms, path):
    from ase.neb import NEB
    
    from ase.rneb import RNEB
    # deleting ions will change inidices
    initial_unrelaxed = atoms.copy()
    del initial_unrelaxed[path[0]]
    
    final_unrelaxed = atoms.copy()
    del final_unrelaxed[path[1]]
    
    # aling indices
    final_unrelaxed = align_indices(initial_unrelaxed, final_unrelaxed)
    images = create_path(initial_unrelaxed, final_unrelaxed)
    neb = NEB(images)
    neb.interpolate()
    
    rneb = RNEB(logfile=None)
    sym = rneb.find_symmetries(atoms, initial_unrelaxed, final_unrelaxed)
    sym = rneb.reflect_path(images, sym=sym)
    print(f"Allowed reflective operations for {path[0]}->{path[1]}: {len(sym)}")
    assert sym is not None # otherwise not reflective
    
def align_indices(initial, final):
    """
    move the atoms in final such that the indices match with initial.

    Parameters
    ---------
    initial: ASE atoms-object

    final: ASE atoms-object

    Return
    -------
    ASE-atoms object
    """

    sort_list = match_atoms(initial, final)    
    final_sorted = final.copy()
    final_sorted = final_sorted[sort_list]

    return final_sorted

def match_atoms(initial, final):
    """
    Match the atomic indices in final to the ones in initial based on
    position

    Parameters
    ---------

    initial: ASE atoms-object
    The initial unrelaxed structure

    final: ASE atoms-object
    The final unrelaxed structure

    Returns
    -------

    sorting list: numpy array 1 x n of integers
    """
    import numpy as np
    pos_initial = initial.get_positions()
    pos_final = final.get_positions()
    symb_initial = initial.get_chemical_symbols()
    symb_final = final.get_chemical_symbols()
    sort_list = np.ones(len(initial)).astype(int)*-1
    no_match = []
    for idx_final, state in enumerate(zip(pos_final, symb_final)):
        pf, sf = state
        idx_initial = find_atom(pos_initial, symb_initial, pf, sf,
                                  critical=False)
        if idx_initial is False:
            no_match.append(idx_final)
        else:        
            sort_list[idx_initial] = idx_final
    missing = np.where(sort_list == -1)  
    if len(missing) != len(no_match):
        if missing[0] != no_match[0]:
            msg = ("different number of atoms have moved "
                   "in the initial and final structures")
            raise RuntimeError(msg)
    
    if len(no_match) > 1:
        msg = ("Found more than one moving atom. "
               "Please give information about where "
               "the moving atoms start and finish")
        raise RuntimeError(msg)
    sort_list[missing[0]] = no_match[0]
    return sort_list

def find_atom(pos, symb, pos_d, symb_d, critical=True, tol=1e-3):
    """
    Find the atom matching pos_d.

    Parameters
    ---------

    pos: numpy array n x 3
    positions to be matched against

    sym: string
    chemical symbols to be matched against

    pos_d: numpy array 1 x 3
    position to be found

    sym_d: string
    chemical symbols to be found

    critical: boolean
    whether to fail upon not being able to make a match

    Return
    -------
    integer if match is found otherwise False: integer or boolean
    """
    import numpy as np
    for i, state in enumerate(zip(pos, symb)):
        p, s = state
        if s == symb_d:
            dist = np.linalg.norm(p-pos_d)
            if dist < tol:
                return i
    if critical:
        msg = ("The speified atoms was not found. "
               "Make sure you given the correct positions and symbols")
        raise RuntimeError(msg)
    else:
        return False

