from ase import Atoms
assert Atoms('X', [(-1e-35, 0, 0)], pbc=True,
             cell=[1, 1, 1]).get_scaled_positions()[0, 0] < 1
