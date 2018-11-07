"""Test suite for TemplateAtoms."""
from ase.clease.template_atoms import TemplateAtoms
from ase.build import bulk

def test_fcc():
    unit_cell = bulk("Cu")
    template_atoms = TemplateAtoms(supercell_factor=27, size=None,
                                   skew_threshold=4, unit_cells=[unit_cell],
                                   unit_cell_ids=[0])
    dims = template_atoms.get_size()
    ref = [(1, 1, 1), (1, 1, 2), (2, 2, 2), (2, 2, 3), (2, 2, 4),
           (2, 2, 5), (2, 3, 3), (2, 3, 4), (3, 3, 3)]
    assert dims == ref


def test_hcp():
    unit_cell = bulk("Mg")
    template_atoms = TemplateAtoms(supercell_factor=27, size=None,
                                   skew_threshold=5, unit_cells=[unit_cell],
                                   unit_cell_ids=[0])
    dims = template_atoms.get_size()
    ref = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (1, 3, 1), (1, 3, 2),
           (1, 4, 1), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4), (2, 2, 5),
           (2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 4, 1), (2, 4, 2),
           (2, 4, 3), (2, 5, 1), (2, 5, 2), (2, 6, 1), (2, 6, 2), (3, 3, 1),
           (3, 3, 2), (3, 3, 3), (3, 4, 1), (3, 4, 2), (3, 5, 1), (3, 6, 1),
           (4, 4, 1), (4, 5, 1)]

    assert dims == ref


test_fcc()
test_hcp()
