# most of these test for DFT being enabled in the control file, as a common
# consequence of failures during define is that defaults such as this won't be
# set, cf. #735
from ase.calculators.turbomole import Turbomole
from reasonablypacedmole import InvalidOccupation, BasisSetNotFound
import ase.io
from functools import partial
import os
import pytest
import re
import shutil

def ase_tm_define(params, define_handler=None):
  if define_handler is not None:
    params = params.copy()
    params["define_handler"] = define_handler
  a = ase.io.read("coord")
  c = Turbomole(**params)
  a.calc = c
  c.initialize()

class AseTm:
  """
  Utility thing for having define_handler separate from the other params and
  accessible from within the test functions
  """
  def __init__(self, define_handler):
    self.define_handler = define_handler
  def define(self, params):
    return ase_tm_define(params, define_handler=self.define_handler)

@pytest.fixture(scope="function")
def ase_tm(request):
  if request.param == "interactive_define":
    define_handler = "interactive"
  else:
    define_handler = None
  return AseTm(define_handler=define_handler)

UNSPEC_MULT_XFAIL = "unspecified multiplicity supported by "\
  "interactive but not by noninteractive define => ASE calc complains"

# hacky shorthand
class file_re:
  def __init__(self, path):
    self.path = path
  def __getattr__(self, attr):
    with open(self.path) as f:
      return partial(getattr(re, attr), string=f.read())
  def find_one(self, *args, **kwargs):
    matches = list(self.finditer(*args, **kwargs))
    assert len(matches) > 0, "found no matches for "\
      f"{args=}, {kwargs=}"
    assert len(matches) < 2, "found more than one match for "\
      f"{args=}, {kwargs=}"
    return matches[0]

h2_coords_contents = """\
$coord
    0.00000000000000      0.00000000000000      0.69652092463935      h
    0.00000000000000      0.00000000000000     -0.69652092463935      h
$user-defined bonds
$end
"""

@pytest.fixture
def tmpdir_cwd_with_h2_coord(tmpdir):
  os.chdir(str(tmpdir))
  with open("coord", "w") as f:
    f.write(h2_coords_contents)

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_title(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "title": "H2test",
    "multiplicity": 1,
  })
  assert file_re("control").find_one(r"\$title\s+H2test\s+")

@pytest.mark.parametrize("ase_tm", [
  pytest.param("static_define", marks=pytest.mark.xfail(reason=UNSPEC_MULT_XFAIL)),
  "interactive_define"
], indirect=True)
def test_h2_guess_mult_otherwise_empty(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": "guess",
  })

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_defaults(tmpdir_cwd_with_h2_coord, ase_tm):
  # XXX kind of the same test as test_h2_explicit_multiplicity below except for
  # checks...
  ase_tm.define({
    "multiplicity": 1,
  })
  # some random checks: DFT on, def-SV(P) chosen as default basis
  control_re = file_re("control")
  assert control_re.search("dft") is not None, "DFT should be on by default"
  assert control_re.find_one(r'\s*basis =[^\n]*def-SV\(P\)')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_ired(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "use redundant internals": True,
    "multiplicity": 1,
  })
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_explicit_basis_set(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "basis set name": "def2-TZVP",
    "multiplicity": 1,
  })
  assert file_re("control").find_one(r'\s*basis =[^\n]*def2-TZVP')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_explicit_multiplicity(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
  })
  assert file_re("control").find_one(r'\$closed shells\n a[ ]+1\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_charged(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "total charge": -1,
    "multiplicity": 2,
  })
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-2\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [
  pytest.param("static_define", marks=pytest.mark.xfail(reason=UNSPEC_MULT_XFAIL)),
  "interactive_define"
], indirect=True)
def test_h2_uhf_guess_mult(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": "guess",
    "uhf": True,
  })
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_uhf(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "uhf": True,
  })
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_triplet(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 3,
  })
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-2\s')
  assert file_re("control").search(r'\$beta shells\n a[ ]+0\s') is None
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [
  pytest.param("static_define", marks=pytest.mark.xfail(reason="static define doesn't support this")),
  "interactive_define"
], indirect=True)
def test_h2_triplet_rohf(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 3,
    "uhf": False,
  })
  assert file_re("control").find_one(r'\$open shells type=1\n a[ ]+1-2\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [
  pytest.param("static_define", marks=pytest.mark.xfail(reason="static define doesn't check this")),
  "interactive_define"
], indirect=True)
def test_h2_uncharged_doublet_that_cant_work(tmpdir_cwd_with_h2_coord, ase_tm):
  with pytest.raises(InvalidOccupation):
    ase_tm.define({
      "multiplicity": 2,
    })
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_damp(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "initial damping": 1.0,
    "damping adjustment step": 2.0,
    "minimal damping": 3.0,
  })
  match = file_re("control").find_one(r'\$scfdamp[ ]+start=[ ]*([0-9.]+)[ ]+step=[ ]*([0-9.]+)[ ]+min=[ ]*([0-9.]+)')
  assert float(match.group(1)) == pytest.approx(1.0)
  assert float(match.group(2)) == pytest.approx(2.0)
  assert float(match.group(3)) == pytest.approx(3.0)
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_scfiter(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "scf iterations": 300,
  })
  with open("control") as f:
    for line in f:
      if line.startswith("$scfiterlimit"):
        if line.split()[1] == "300":
          return
    assert False, "no scfiterlimit found in control file!"
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_scfconv(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "scf energy convergence": 1e-5,
  })
  with open("control") as f:
    for line in f:
      if line.startswith("$scfconv"):
        if line.split()[1] == "6": # XXX this is what it does, but why not 5?
          return
    assert False, "no scfconv found in control file!"
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_fermi(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "use fermi smearing": True,
    "fermi initial temperature": 400,
    "fermi final temperature": 200,
    "fermi annealing factor": 0.9,
    "fermi homo-lumo gap criterion": 0.08,
    "fermi stopping criterion": 0.004,
  })
  # $fermi tmstrt=400.00 tmend=200.00 tmfac=0.900 hlcrt=8.0E-02 stop=4.0E-03
  with open("control") as f:
    for line in f:
      if line.startswith("$fermi"):
        d = {}
        for assignment in line.split()[1:]:
          lhs, rhs = assignment.split("=")
          d[lhs] = float(rhs)
    assert d["tmstrt"] == pytest.approx(400)
    assert d["tmend"] == pytest.approx(200)
    assert d["tmfac"] == 0.9
    assert d["hlcrt"] == 0.08
    assert d["stop"] == 0.004
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_dft(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "use dft": True,
    "density functional": "b3-lyp",
  })
  with open("control") as f:
    found_dft_block, found_functional = False, False
    for line in f:
      if line.startswith("$dft"):
        found_dft_block = True
        continue
      if line.strip().startswith("functional"):
        assert line.split()[1] == "b3-lyp"
        found_functional = True
        break
    assert found_dft_block
    assert found_functional

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_dft_ri(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "use dft": True,
    "density functional": "b3-lyp",
    "use resolution of identity": True,
  })
  assert file_re("control").find_one(r'\$dft\n')
  assert file_re("control").find_one(r'[ ]+functional[ ]+b3-lyp\n')
  assert file_re("control").find_one(r'\$ricore')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_dft_grid_size(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "use dft": True,
    "grid size": "m4",
  })
  assert file_re("control").find_one(r'\$dft\n')
  assert file_re("control").find_one(r'[ ]+gridsize[ ]+m4\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_dft_ri_mem(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "use dft": True,
    "density functional": "b3-lyp",
    "use resolution of identity": True,
    "ri memory": 1000,
  })
  assert file_re("control").find_one(r'\$dft\n')
  assert file_re("control").find_one(r'[ ]+functional[ ]+b3-lyp\n')
  assert file_re("control").find_one(r'\$ricore[ ]+1000\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_fails_with_invalid_basis_set(tmpdir_cwd_with_h2_coord, ase_tm):
  if ase_tm.define_handler == "interactive":
    expected_exception = BasisSetNotFound
  else:
    expected_exception = Exception
  with pytest.raises(expected_exception):
    ase_tm.define({
      "multiplicity": 1,
      "basis set name": "defgarbage-SVgarbage",
    })

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_define_str(tmpdir_cwd_with_h2_coord, ase_tm):
  define_str = '\n\na coord\n*\nno\n*\neht\n\n\n\n*\n'
  ase_tm.define({
    "define_str": define_str,
  })

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_define_str_without_final_newline(tmpdir_cwd_with_h2_coord, ase_tm):
  define_str = '\n\na coord\n*\nno\n*\neht\n\n\n\n*'
  ase_tm.define({
    "define_str": define_str,
  })

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_hcore_guess(tmpdir_cwd_with_h2_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 1,
    "initial guess": "hcore",
  })
  assert file_re("control").find_one(r'\$scfmo none')
  assert file_re("control").find_one(r'\$dft\n') # extra check

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_h2_use_guess(tmpdir_cwd_with_h2_coord, ase_tm):
  # first, define calculation with standard EHT guess but non-default occ
  ase_tm.define({
    "total charge": -1,
    "multiplicity": 2,
    "uhf": True,
  })
  # move this to separate directory
  os.mkdir("previous")
  for x in ["control", "alpha", "beta", "basis"]:
    shutil.move(x, "previous")
  # now define new calculation with different occ but previous one as starting
  # guess
  ase_tm.define({
    "multiplicity": 3,
    "uhf": True,
    "initial guess": { "use": "previous/control" },
  })
  # check that occupation matches the one from the starting guess
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-2\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1\s')
  assert file_re("control").find_one(r'\$dft\n') # extra check

au13_coords_contents = """\
$coord
    4.63762204312598      4.63762204312598      4.63762204312598      au
    9.27524408625195      4.63762204312598      1.77141399349839      au
    9.27524408625195      4.63762204312598      7.50383009275356      au
    0.00000000000000      4.63762204312598      1.77141399349839      au
    0.00000000000000      4.63762204312598      7.50383009275356      au
    1.77141399349839      9.27524408625195      4.63762204312598      au
    7.50383009275356      9.27524408625195      4.63762204312598      au
    1.77141399349839      0.00000000000000      4.63762204312598      au
    7.50383009275356      0.00000000000000      4.63762204312598      au
    4.63762204312598      1.77141399349839      9.27524408625195      au
    4.63762204312598      7.50383009275356      9.27524408625195      au
    4.63762204312598      1.77141399349839      0.00000000000000      au
    4.63762204312598      7.50383009275356      0.00000000000000      au
$end
"""
@pytest.fixture
def tmpdir_cwd_with_au13_coord(tmpdir):
  os.chdir(str(tmpdir))
  with open("coord", "w") as f:
    f.write(au13_coords_contents)

@pytest.mark.parametrize("ase_tm", [
  pytest.param("static_define", marks=pytest.mark.xfail(reason="static define doesn't check this")),
  "interactive_define"
  ], indirect=True)
def test_au13_guess_mult(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "multiplicity": "guess",
  })
  # default mult is 6
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-126\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1-121\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_au13_mult6_same_as_default(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 6,
  })
  # just check that it's unchanged
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-126\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1-121\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_au13_mult4(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "multiplicity": 4,
  })
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-125\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1-122\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_au13_chargedm1_mult5_same_as_default(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "total charge": -1,
    "multiplicity": 5,
  })
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-126\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1-122\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_au13_chargedm1_mult1_rohf(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "total charge": -1,
    "multiplicity": 1,
    "uhf": False,
  })
  assert file_re("control").find_one(r'\$closed shells\n a[ ]+1-124\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_au13_chargedm1_mult3(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "total charge": -1,
    "multiplicity": 3,
  })
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-125\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1-123\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_au13_chargedm1_mult7(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "total charge": -1,
    "multiplicity": 7,
  })
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-127\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1-121\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.xfail(reason="roothan parameter thing - don't know what to do")
@pytest.mark.skip(reason="no point testing this until correct result known")
def test_au13_chargedm1_mult7_rohf(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "total charge": -1,
    "multiplicity": 7,
    "uhf": False,
  })
  assert file_re("control").find_one(r'\$dft\n')
  # TODO add actual checks here

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
def test_au13_chargedp1_mult5_same_as_default(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "total charge": 1,
    "multiplicity": 5,
  })
  assert file_re("control").find_one(r'\$alpha shells\n a[ ]+1-125\s')
  assert file_re("control").find_one(r'\$beta shells\n a[ ]+1-121\s')
  assert file_re("control").find_one(r'\$dft\n')

@pytest.mark.parametrize("ase_tm", [ "static_define", "interactive_define" ],
  indirect=True)
@pytest.mark.xfail(reason="roothan parameter thing - don't know what to do")
@pytest.mark.skip(reason="no point testing this until correct result known")
def test_au13_chargedp1_mult5_rohf(tmpdir_cwd_with_au13_coord, ase_tm):
  ase_tm.define({
    "total charge": 1,
    "multiplicity": 5,
    "uhf": False,
  })
  assert file_re("control").find_one(r'\$dft\n')
  # TODO add actual checks here
