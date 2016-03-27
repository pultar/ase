from ase.lattice.surface import fcc111
from ase.lattice.root import root_surface

# Generate common example in literature
sub = fcc111("Pt", (1, 1, 3), 3.9, vacuum=8)
film = fcc111("Fe", (1, 1, 2), 2.0)
film[1].symbol = "O"

# Use 67 and 84 for a system closer to literature
sub = root_surface(sub, 3)
film = root_surface(film, 4)

interface = sub.stack(film, 3)
# interface.edit()

assert(len(interface) == len(sub) + len(film))
assert(interface[0].position[2] < interface[-1].position[2])
