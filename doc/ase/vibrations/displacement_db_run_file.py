import os
from pathlib import Path
import shutil

from ase.calculators.espresso import Espresso, EspressoProfile
from ase.calculators.singlepoint import SinglePointCalculator
import ase.db


def main():
    # Get job array index from batch scheduler
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print(f"Running array task {task_id}")
    run_dir = f'disp-{task_id}'

    with ase.db.connect('displacements.db') as db:
        row = next(db.select(id=task_id))
        xc = row.get('xc')
        name = row.get('name')
        label = row.get('label')

        displacement = row.toatoms()

    print(f"Got DB row {task_id}: {name} {xc} {label}")

    calc = Espresso(
        directory=run_dir,
        profile=EspressoProfile(['mpirun', 'pw.x']),
        pseudo_dir=str(Path.home()
                       / 'opt/pseudopotentials/sssp_1.1.2_PBE_precision'),
        pseudopotentials={'C': 'C.pbe-n-kjpaw_psl.1.0.0.UPF',
                          'O': 'O.pbe-n-kjpaw_psl.0.1.UPF',
                          'H': 'H_ONCV_PBE-1.0.oncvpsp.upf'},
        input_data={'control': {'calculation': 'scf',
                                'tprnfor': True},  # Calculate forces
                    'electrons': {'conv_thr': 1e-8},
                    'system': {'input_dft': xc}})

    displacement.calc = calc
    forces = displacement.get_forces()

    # Use a lightweight calculator to store force data and update DB row
    with ase.db.connect('displacements.db') as db:
        displacement_copy = displacement.copy()
        displacement_copy.calc = SinglePointCalculator(atoms=displacement_copy,
                                                       forces=forces)

        db.update(id=task_id, atoms=displacement_copy)

    # If we didn't hit an error yet, remove the calculation directory
    shutil.rmtree(run_dir)


if __name__ == '__main__':
    main()
