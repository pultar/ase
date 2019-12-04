from numpy.testing import assert_allclose

from ase.build import bulk
from ase.calculators.nwchem import NWChem
from ase.dft.band_structure import calculate_band_structure

atoms = bulk('C')
atoms.calc = NWChem(label='diamond_band_structure', kpts=(9, 9, 9),
                    symmetry=227, nwpw=dict(virtual=4))
bs = calculate_band_structure(atoms)

#assert_allclose(bs.energies[0, 0],
#                [-8.24774389, 12.96198046, 12.96203489, 12.96208931],
#                atol=1e-4, rtol=1e-4)
#
#assert_allclose(bs.energies[0, -1],
#                [0.45510227, 0.45535533, 6.72668184, 6.72668184],
#                atol=1e-4, rtol=1e-4)
