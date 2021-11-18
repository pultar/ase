# Interactive user

from gpaw import GPAW
from ase.vibrations import get_displacements, read_forces_strict
import ase.io

atoms = ase.io.read('my_file.cif')

options = dict(delta=0.01, nfree=2,
               indices=range(len(atoms))[atoms.get_symbols()=='H'])

calc = GPAW()
displacements = get_displacements(atoms, **options)

for disp in displacements:
    disp.calc = calc
    _ = disp.get_forces()

vibs = read_forces_strict(
    displacements, **options)

vibs.write_json('my_vibrations.json')

##############################################################
## Batch user
import ase.io
from ase.vibrations import write_displacements

atoms = ase.io.read('my_file.cif')

options = dict(delta=0.01, nfree=2,
               indices=range(len(atoms))[atoms.get_symbols()=='H'])

import json
with open('my_options.json', 'w') as fd:
    json.dump(options, fd)

write_displacements(atoms, format='aims', name='vibs', **options)

## Go away and run calcs for structures inside vibs
## In a new python session:

import globbing
import json
import ase.io
from ase.vibrations import read_forces_strict

atoms = ase.io.read('my_file.cif')

with open('my_options.json', 'r') as fd:
    options = json.load(fd)

displacements = [ase.io.read(filename, format='aims_out')
                 for filename in globbing.glob('vibs/*/*.out')]                 ]

vibs = read_forces_strict(displacements, **options)

##################################################################
## Database user
import ase.db
from ase.vibrations import write_displacements_to_db

atoms = ase.io.read('my_file.cif')

options = dict(delta=0.01, nfree=2,
               indices=[1, 2, 3, 4])

with ase.db.connect('mydb.db') as db:
    write_displacements_to_db(atoms, db=db,  name='system-1',
                              metadata={'cheese': 'gouda'},
                              **options)

## During a cluster job:
import ase.db
from gpaw import GPAW()

with ase.db.connect('mydb.db') as db:
    disp_row = next(db.select(name='system-1',
                              batch_number=SLURM_TASK_ID))
    displacement = disp_row.to_atoms()

dispacement.calc = GPAW()
displacement.get_forces()

with ase.db.connect('mydb.db') as db:
    db.write(displacement, name='system-1', cheese=disp_row.cheese)

### Post-process

with ase.db.connect('mydb.db') as db:
    options = dict(delta=0.01, nfree=2,
                   indices=[1, 2, 3, 4])
    
    vibs = read_forces_from_db(db, **options, name='system-1')
