"""Test suite for TemplateAtoms."""
from ase.clease.template_atoms import TemplateAtoms
from ase.build import bulk


def test_fcc():
    unit_cell = bulk("Cu")
    template_atoms = TemplateAtoms(unit_cells=[unit_cell], skew_threshold=4,
                                   supercell_factor=27)
    dims = template_atoms.get_dims()
    ref = [(1, 1, 1), (1, 1, 2), (2, 2, 2), (2, 2, 3), (2, 2, 4),
           (2, 2, 5), (2, 3, 3), (2, 3, 4), (3, 3, 3)]
    assert dims == ref

def test_hcp():
    unit_cell = bulk("Mg")
    template_atoms = TemplateAtoms(unit_cells=[unit_cell], skew_threshold=5,
                                   supercell_factor=27)
    dims = template_atoms.get_dims()
    ref = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (1, 3, 1), (1, 3, 2),
           (1, 4, 1), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4), (2, 2, 5),
           (2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 4, 1), (2, 4, 2),
           (2, 4, 3), (2, 5, 1), (2, 5, 2), (2, 6, 1), (2, 6, 2), (3, 3, 1),
           (3, 3, 2), (3, 3, 3), (3, 4, 1), (3, 4, 2), (3, 5, 1), (3, 6, 1),
           (4, 4, 1), (4, 5, 1)]

    assert dims == ref


test_fcc()
test_hcp()


# If user does not specify primitive/cubic --> both
# If user specify size primtivie/cubic has to be specified
# When settings are reconfigured then check if entries already in the database is consistent with a template atoms object, if not set a flag saying that this structure should be excluded
# Add function for deleting excluded structures from the database.
# In database information entries should have a key-value-pair with the repeated dimension i.e. 1x1x3,
# Skewness factors should also be stored in the database
