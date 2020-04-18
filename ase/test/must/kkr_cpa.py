from ase.test import require
from ase.calculators.must import MuST, generate_starting_potentials
from ase.build import bulk
import os


def kkr():
    require('must')
    atoms = bulk('Al', a=4.05, cubic=False)
    generate_starting_potentials(atoms, crystal_type=1, a=4.05)
    calc = MuST(default_in_pot='Al_ss_pot',
                default_out_pot='Al',
                pot_in_form=0,
                pot_out_form=0,
                nscf=50,
                method=2,
                potential_type=0,
                xc='GGA_X_PBE_SOL+GGA_C_PBE_SOL',
                spin=1,
                kpts=(12, 12, 12),
                bzsym=1)

    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    ref = -6582.117008139085
    assert (energy - ref) / ref < 1e-10
    print('Passed KKR test')

    # Clean up
    calc.clean()

    files = ['Al_a_in', 'Al_a_out', 'Al_a_pot', 'Al_ss_in',
             'Al_ss_out', 'Al_ss_k', 'Al_ss_pot', 'Al_0001x1']
    for f in files:
        os.remove(f)


def kkr_cpa():
    atoms = bulk('Al', 'bcc', a=2.861, cubic=False)
    atoms.info['CPA'] = [{'index': 0, 'Al': 0.2, 'Cr': 0.2,
                          'Fe': 0.2, 'Co': 0.2, 'Ni': 0.2}]
    generate_starting_potentials(atoms, crystal_type=2, a=2.861, cpa=True)
    calc = MuST(default_in_pot='Al_ss_pot Cr_ss_pot Fe_ss_pot' +
                               'Co_ss_pot Ni_ss_pot',
                default_out_pot='AlCrFeCoNi',
                pot_in_form=0,
                pot_out_form=1,
                nscf=80,
                method=3,
                potential_type=0,
                xc='GGA_X_PBE_SOL+GGA_C_PBE_SOL',
                spin=1,
                kpts=(12, 12, 12),
                bzsym=1)
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    ref = -29585.112262706632
    assert (energy - ref) / ref < 1e-10
    print('Passed KKR-CPA test')

    # Clean up
    calc.clean()

    for el in atoms.info['CPA'][0].keys():
        if el == 'index':
            pass
        else:
            files = [el + '_a_in', el + '_a_out', el + '_a_pot', el + '_ss_in',
                     el + '_ss_out', el + '_ss_k', el + '_ss_pot', ]
            for f in files:
                os.remove(f)

    os.remove('AlCrFeCoNi')


kkr()
kkr_cpa()
