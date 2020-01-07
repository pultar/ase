import ase.db
from ase import Atoms
import numpy as np

for name in ['x.json', 'x.db']:
    print(name)
    db = ase.db.connect(name, append=False)
    atoms = Atoms('H2')
    atoms.arrays['labels'] = np.array(['H1', 'H2'], dtype=object)
    db.write(atoms, data={'a': 1})
    row = db.get(1)
    print(row.data)
    assert '_custom_arrays' in row.data
    at = row.toatoms()
    assert np.all(at.arrays['labels'] == np.array(['H1', 'H2']))
    atoms = Atoms('H3')
    atoms.arrays['labels'] = np.array(['H1', 'H2', 'H3'], dtype=object)
    db.update(1, atoms=atoms, data={'b': 2})
    row = db.get(1)
    print(row.data, row.numbers)
    at = row.toatoms(add_additional_information=True)
    assert np.all(at.arrays['labels'] == np.array(['H1', 'H2', 'H3']))
    assert sorted(at.info['data']) == ['a', 'b']

    db.write(Atoms(), id=1)
    row = db.get(1)
    assert len(row.data) == 0
    assert len(row.key_value_pairs) == 0
    assert len(row.numbers) == 0
