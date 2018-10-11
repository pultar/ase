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
    counter = {}
    for _ in range(10000):
        atoms, dim = template_atoms.weighted_random_template(return_dim=True)
        if dim not in counter.keys():
            counter[dim] = 1
        else:
            counter[dim] += 1
    print(counter)

test_fcc()
