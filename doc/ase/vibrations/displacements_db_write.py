import ase.build
from ase.calculators.espresso import Espresso, EspressoProfile
import ase.io
from ase.vibrations.finite_diff import write_displacements_to_db
from pathlib import Path

atoms = ase.build.molecule('CH3CH2OH', vacuum=7.)

for xc in ('PBE', 'revPBE'):
    calc = Espresso(
        directory=f'opt-{xc}',
        profile=EspressoProfile(['mpirun', 'pw.x']),
        pseudo_dir=str(Path.home()
                       / 'opt/pseudopotentials/sssp_1.1.2_PBE_precision'),
        pseudopotentials={'C': 'C.pbe-n-kjpaw_psl.1.0.0.UPF',
                          'O': 'O.pbe-n-kjpaw_psl.0.1.UPF',
                          'H': 'H_ONCV_PBE-1.0.oncvpsp.upf'},
        input_data={'control': {'calculation': 'relax',
                                'forc_conv_thr': 2e-5,
                                'nstep': 100},
                    'electrons': {'conv_thr': 1e-8},
                    'system': {'input_dft': xc}})

    # Optimise geometry using Espresso internal optimiser
    atoms.calc = calc
    atoms.get_forces()

    opt_atoms = ase.io.read(f'opt-{xc}/espresso.pwo')
    opt_atoms.write(f'ethanol-{xc}.extxyz')

    write_displacements_to_db(opt_atoms, metadata={'name': 'ethanol',
                                                   'xc': xc})
