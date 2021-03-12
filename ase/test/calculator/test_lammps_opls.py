import pytest

from ase.io.opls import OPLSStructure
from ase.calculators.oplslmp import OPLSlmp
from ase.optimize import BFGS


@pytest.fixture
def H2():
    structure = 'H2.extxyz'
    with open(structure, 'w') as f:
        f.write("""2
Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3:molid:I:1:type:S:1 pbc="T T T"
H 0.5 0 0 1 HH
H 0 0 0 1 HH
""")
    return OPLSStructure(structure)


@pytest.mark.calculator_lite
@pytest.mark.calculator('lammpsrun')
def test_H2_opls(factory, H2):
    params = 'params.par'
    with open(params, 'w') as f:
        f.write("""# one body
HH 0.0001 0.0001 0

# Bonds
HH-HH 4401.21 0.74144

# Angles

# Dihedrals

# Cutoffs
HH-HH 0.8  # force "bond breaking" during relaxation
""")

    cmd = factory.calc().parameters.get('command')
    H2.calc = OPLSlmp(params, command=cmd)
    opt = BFGS(H2)
    opt.run(fmax=0.01)

    assert H2.get_distance(0, 1) == pytest.approx(0.74144, 1e-4)


@pytest.mark.calculator_lite
@pytest.mark.calculator('lammpsrun')
def test_C6H12O2(factory, datadir):
    atoms = OPLSStructure(datadir / 'C6H12O2_opls.extxyz')
    cmd = factory.calc().parameters.get('command')
    atoms.calc = OPLSlmp(datadir / 'C6H12O2_opls.par', command=cmd)

    opt = BFGS(atoms)
    opt.run(fmax=0.1)

    assert atoms.get_distance(3, 19) == pytest.approx(0.95807, 1e-3)
