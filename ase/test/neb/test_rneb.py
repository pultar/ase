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
