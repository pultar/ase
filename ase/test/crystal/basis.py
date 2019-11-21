from ase.calculators.crystal import CRYSTAL
from ase.build import bulk

# NaCl Fm-3m ICSD 240598
a = 5.6401
bulk = bulk('NaCl', 'rocksalt', a=a)

calc = CRYSTAL(label='sodium chloride',
              guess=True,
              basis='sto-3g',
              xc='PBE',
              otherkeys=['scfdir', 'anderson',
                        ['maxcycles', '100'],
                        ['toldee', '5'],
                        ['tolinteg', '7 7 7 7 14'],
                        ['fmixing', '90']])

bulk.set_calculator(calc)
final_energy = bulk.get_potential_energy()
assert abs(final_energy + 16741.0204850684) < 1.0

