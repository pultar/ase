import abc
import typing as tp
import numpy as np
from ase.atoms import Atoms


DEFAULT_DELTA = 1e-2


class Displacement(tp.Protocol):
    """API for a displacement.

    A Displacement is a hashable representation of a single atomic displacement (e.g.
    "atom 3 in the +z direction"), which additionally provides a name for use in files.
    """
    @property
    def name(self) -> str:
        """A string that identifies this displacement.  E.g. "3z+".

        This is intended for use in filenames, and also as a hashable key for dicts and sets."""

    @property
    def atom(self) -> int:
        """The index of the atom to be displaced."""

    def vector(self, delta: float) -> np.ndarray:
        """The cartesian 3-vector by which the atom should be displaced.

        This may be zero, for the "equilibrium" displacement.

        Parameters
        ----------
        delta:
            Step size. The returned vector will typically have a magnitude equal to delta,
            or to an integer multiple of delta.
        """


class EqDisplacement:
    @property
    def name(self) -> str:
        return 'eq'

    @property
    def atom(self) -> int:
        return 0

    def vector(self, delta: float) -> np.ndarray:
        return np.zeros((3,), dtype=float)


class Displacements(abc.ABC):
    """Interface for deciding the set of displacements to use when computing the
    first derivative of a physical property with respect to atom positions.

    In the simple case, displacements can simply be chosen to perform an
    independent finite difference stencil along each axis; to do this,
    one can use AxisAlignedDisplacements.

    Other implementations may be defined to do something more sophisticated,
    such as apply symmetry to compute fewer datapoints."""

    def __init__(self, delta: float):
        super().__init__()
        self._delta = delta

    @property
    def delta(self):
        """Get the step size."""
        return self._delta

    # Required methods

    @abc.abstractmethod
    def nontrivial_disps(self) -> tp.Iterator[Displacement]:
        """Iterates over non-equilibrium displacements at which quantities must be computed."""
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_cartesian_derivatives(self, data: np.ndarray):
        """Compute the first derivative of a function, given the values at each displacement.

        Arguments:

        data:
            Values at each displacement, as an ``(ndisp, ...axes)``-shape array where
            ``data[i]`` contains the output of a function computed at the ``i``th
            displacement produced by ``iterdisps``.

        Returns:
            An ``(natom, 3, ...axes)``-shape array ``out`` where ``out[a][i]`` is the
            derivative of the function with respect to the motion of atom ``a`` along the
            ``i``th cartesian axis.
        """
        raise NotImplementedError()

    # Helper methods / Public API

    def eq_disp(self) -> Displacement:
        return EqDisplacement()

    def __iter__(self) -> tp.Iterator[Displacement]:
        """Iterate over all displacements at which to compute data.

        The first displacement is always the equilibrium displacement, even if this data is not
        necessary for computing the first derivative."""
        yield self.eq_disp()
        yield from self.nontrivial_disps()

    def iter_with_atoms(self, initial_atoms: Atoms, inplace=False) -> tp.Iterator[tp.Tuple[Displacement, Atoms]]:
        """Iterate over all displacements and the corresponding atoms.

        Parameters
        ----------
        initial_atoms: Atoms
            The atoms in their equilibrium positions.
        inplace: bool
            By default, the atoms returned are copies, and thus will lack their calculator.
            To reuse a single Atoms through the entire process (and thus keep its calculator),
            use ``inplace=True``.
        """
        atoms = initial_atoms if inplace else initial_atoms.copy()
        displacements = iter(self)
        eq_disp = next(displacements)
        assert eq_disp.name == 'eq'
        yield eq_disp, atoms

        for disp in displacements:
            if not inplace:
                atoms = initial_atoms.copy()
            pos0 = atoms.positions[disp.atom].copy()
            atoms.positions[disp.atom] += disp.vector(self.delta)
            yield disp, atoms

            if inplace:
                atoms.positions[disp.atom] = pos0

    # Optional methods

    # FIXME: I am worried that one thing could call this while another thing actually does need all
    # indices.  Maybe instead want a "require_indices" that does all atoms if never called, but does the
    # union of atoms if it is called at least once...?  (or remove that last exception and expect everything
    # to call it?)
    def for_atom_indices(self, indices: tp.Iterable[int]) -> 'Displacements':
        """Provide a hint to this Displacements that we are only interested in the derivatives
        with respect to the given atoms.  This will not modify `self`.

        Instances of Displacements MAY use this information to reduce the number of displacements they
        generate.  More precisely, calling this method enables the Displacements to return arbitrary
        data (e.g. NaN or 0) for the derivatives with respect to atoms outside the given list.
        (however, implementations are not required to do anything at all!)

        FIXME TODO CODE EXAMPLE
        """
        return self  # "do nothing" is a valid implementation of this method


class AxisAlignedDisplacementBase(tp.NamedTuple):
    atom: int
    axis: int
    step: int


class AxisAlignedDisplacement(AxisAlignedDisplacementBase):
    @property
    def name(self) -> str:
        sign = '+' if self.step > 0 else '-'
        multiple = '' if abs(self.step) == 1 else str(self.step)
        return f'{self.atom}{"xyz"[self.axis]}{sign}{multiple}'

    def vector(self, delta: float) -> np.ndarray:
        direction = np.zeros((3,), dtype=float)
        direction[self.axis] = 1
        return direction * delta * self.step

    def __hash__(self):
        raise TypeError(f'{type(self).__name__} is not hashable, please use disp.name')


# TODO: tests that each stencil is accurate up to expected error
# TODO: test that non-displaced means zero derivative
class AxisAlignedDisplacements(Displacements):
    """Displacements for computing the derivative via a simple stencil
    method on each cartesian axis.

    Arguments:

    nfree:
        Number of points in addition to the center point along each axis.
        E.g. ``nfree=4`` for a 5-point stencil.
    direction:
        Finite-difference scheme. ``'central'``, ``'forward'``, or ``'backward'``
    indices:
        Only displace the chosen atoms.  The derivatives with respect
        to other atoms will be to zero.
    """

    STENCIL_DATA = {
        ('central', 2): {'points': [1, -1], 'coeffs': [1, -1], 'divisor': 2},
        ('forward', 2): {'points': [1, 0], 'coeffs': [1, -1], 'divisor': 1},
        ('backward', 2): {'points': [0, -1], 'coeffs': [1, -1], 'divisor': 1},
        ('central', 4): {'points': [2, 1, -1, -2], 'coeffs': [-1, 8, -8, 1], 'divisor': 12},
    }

    def __init__(self, atoms, *,
                 indices: tp.Optional[tp.Sequence[int]] = None,
                 nfree: int = 2,
                 direction: str = 'central',
                 delta: float = DEFAULT_DELTA):
        super().__init__(delta=delta)

        direction = direction.lower()
        assert nfree in [2, 4]
        assert direction in ['central', 'backward', 'forward']

        if indices is None:
            indices = range(len(atoms))
        if len(indices) != len(set(indices)):
            raise ValueError(
                'one (or more) indices included more than once')

        if (direction, nfree) not in self.STENCIL_DATA:
            raise RuntimeError(
                f'stencil for direction={repr(direction)}, nfree={nfree} not implemented')

        self._natom = len(atoms)
        self._indices = indices
        self._nfree = nfree
        self._direction = direction
        self._stencil = self.STENCIL_DATA[self._direction, self._nfree]

    def nontrivial_disps(self):
        steps = [x for x in self._stencil['points'] if x != 0]

        for atom in self._indices:
            for axis in range(3):
                for step in steps:
                    yield AxisAlignedDisplacement(atom=atom, axis=axis, step=step)

    def _get_disp(self, atom, axis, step):
        if step == 0:
            return self.eq_disp()
        else:
            return AxisAlignedDisplacement(atom=atom, axis=axis, step=step)

    def compute_cartesian_derivatives(self, data: np.ndarray):
        disp_indices = {disp.name: i for (i, disp) in enumerate(self)}
        if len(disp_indices) != len(data):
            raise ValueError(f"data has wrong size for outermost dimension (expected {len(disp_indices)}, got {len(data)})")

        coeffs = np.array(self._stencil['coeffs'])
        steps = np.array(self._stencil['points'])

        derivs = np.zeros((self._natom, 3) + data.shape[1:], dtype=data.dtype)
        for atom in self._indices:
            for axis in range(3):
                derivs[atom][axis] = sum(
                    coeff * data[disp_indices[self._get_disp(atom=atom, axis=axis, step=step).name]]
                    for (coeff, step) in zip(coeffs, steps)
                ) / (self._stencil['divisor'] * self.delta)

        return derivs
