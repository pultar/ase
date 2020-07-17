from ase import Atoms
from ase.data.pubchem import pubchem_atoms_search
from ase.io import write

cn = pubchem_atoms_search(name='cyanide')
cn.translate(-cn[1].position)  # center carbon in origo
cn.translate([2, 0, 0])  # distance from Fe in Origo

liga = Atoms()
new = cn.copy()
for i in range(3):
    new.rotate(90, 'z')
    liga += new
ligb = liga.copy()
ligb.rotate(90, 'x')
ligb.rotate(180, 'y')

ligands = liga + ligb

atoms = Atoms('Fe', positions=[[0, 0, 0]]) + ligands

write('FeCN6.xyz', atoms)
