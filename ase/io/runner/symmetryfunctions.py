"""Classes and routines for reading and storing RuNNer symmetry functions.

The RuNNer Neural Network Energy Representation is a framework for the
construction of high-dimensional neural network potentials developed in the
group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.

Provides
--------

calc_radial_symfuns : utility function
    Return a list of RuNNer radial symmetry functions.
calc_angular_symfuns : utility function
    Return a list of RuNNer angular symmetry functions.

Runner : FileIOCalculator
    The main calculator for training and evaluating HDNNPs with RuNNer.

Reference
---------
* [The online documentation of RuNNer](https://theochem.gitlab.io/runner)

Contributors
------------
* Author: [Alexander Knoll](mailto:alexander.knoll@chemie.uni-goettingen.de)

"""

from typing import Optional, Iterator, Union

from itertools import combinations_with_replacement, product


class SymmetryFunction:
    """Generic class for one single symmetry function."""

    def __init__(
        self,
        sftype: int,
        cutoff: float,
        elements: Optional[list[str]] = None,
        coefficients: Optional[list[float]] = None
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        sftype : int
            The type of symmetry function.
        cutoff : float
            The symmetry function cutoff radius in units of Bohr.

        Optional Arguments
        ------------------
        coefficients : list[float]
            The coefficients of this symmetry function. The number of necessary
            coefficients depends on the `sftype` as documented in the RuNNer
            manual.
        elements : list[str]
            Symbols of the elements described by this symmetry function.
            The first entry of the list gives the central atom, while all
            following entries stand for neighbor atoms. Usually, the number of
            neighbors should be 1 (= radial symfun) or 2 (= angular symfun)
        """
        self.sftype = sftype
        self.cutoff = cutoff
        self.elements = elements
        self.coefficients = coefficients

    def __repr__(self) -> str:
        """Show a clean summary of the class object and its contents."""
        return f'{self.__class__.__name__}(type={self.sftype}, ' \
             + f'cutoff={self.cutoff}, ' \
             + f'elements={self.elements}, coefficients={self.coefficients})'

    def __add__(
        self,
        blob: Union['SymmetryFunctionSet', 'SymmetryFunction']
    ) -> Union['SymmetryFunctionSet', 'SymmetryFunction']:
        """Define addition behaviour."""
        if isinstance(blob, SymmetryFunctionSet):
            blob += self
            return blob

        if isinstance(blob, SymmetryFunction):
            sfset = SymmetryFunctionSet()
            sfset += self
            sfset += blob
            return sfset

        raise TypeError('unsupported operand type(s) for multiplication.')

    def __mul__(self, multiplier: int) -> 'SymmetryFunctionSet':
        """Define multiplication behaviour."""
        sfset = SymmetryFunctionSet()
        for _ in range(multiplier):
            sfset += self.copy()

        return sfset

    def __rmul__(self, multiplier: int) -> 'SymmetryFunctionSet':
        """Multiply the object from the left."""
        return self.__mul__(multiplier)

    def copy(self) -> 'SymmetryFunction':
        """Make the object copyable."""
        return SymmetryFunction(self.sftype, self.cutoff, self.elements,
                                self.coefficients)

    @property
    def tag(self) -> str:
        """Show a human-readable equivalent of the `sftype` parameter."""
        if self.sftype in [2]:
            tag = 'radial'
        elif self.sftype in [3, 4, 5]:
            tag = 'angular'
        else:
            tag = 'unknown'

        return tag

    def to_runner(self, fmt: str = '16.8f') -> str:
        """Convert the symmetry function into RuNNer input.nn format."""
        if self.coefficients is None or self.elements is None:
            raise Exception('Symmetryfunction is not fully defined yet.')

        centralatom = self.elements[0]
        neighbors = self.elements[1:]
        coefficients = [f'{c:{fmt}}' for c in self.coefficients]

        string = f'{centralatom} {self.sftype} ' \
                + ' '.join(neighbors) \
                + ' '.join(coefficients) \
                + f' {self.cutoff:{fmt}}'

        return string


class SymmetryFunctionSet:
    """Class for storing groups/sets of symmetry functions."""

    def __init__(
        self,
        min_distances: Optional[dict[str, float]] = None
    ) -> None:
        """Initialize the class.

        This class can be nested to group symmetry functions together.

        Optional Parameters
        -------------------
        elements : list[str], _default_ None
        min_distances : list[float], _default_ None
        """
        self._sets: list[SymmetryFunctionSet] = []
        self._symmetryfunctions: list[SymmetryFunction] = []
        self.min_distances = min_distances

    def __repr__(self) -> str:
        """Show a clean summary of the class object and its contents."""
        n_sets = len(self._sets)
        n_symmetryfunctions = len(self._symmetryfunctions)

        return f'{self.__class__.__name__}(type={self.sftypes}, ' \
             + f'sets={n_sets}, symmetryfunctions={n_symmetryfunctions})'

    def __len__(self) -> int:
        """Return the number of symmetry functions as the object length."""
        return len(self._symmetryfunctions)

    def __add__(
        self,
        blob: Union['SymmetryFunctionSet', SymmetryFunction]
    ) -> 'SymmetryFunctionSet':
        """Overload magic routine to enable nesting of multiple sets."""
        # Add a new subset of symmetry functions.
        if isinstance(blob, SymmetryFunctionSet):
            self._sets += blob.sets
            self._symmetryfunctions += blob.symmetryfunctions

        # Add a single symmetry function to storage.
        elif isinstance(blob, SymmetryFunction):
            self._symmetryfunctions.append(blob)
        else:
            raise NotImplementedError

        return self

    def __iter__(self) -> Iterator[SymmetryFunction]:
        """Iterate over all stored symmetry functions."""
        for symmetryfunction in self.storage:
            yield symmetryfunction

    @property
    def sets(self) -> list['SymmetryFunctionSet']:
        """Show a list of all stored `SymmetryFunctionSet` objects."""
        return self._sets

    @property
    def symmetryfunctions(self) -> list[SymmetryFunction]:
        """Show a list of all stored `SymmetryFunction` objects."""
        return self._symmetryfunctions

    @property
    def storage(self) -> list[SymmetryFunction]:
        """Show all stored symmetry functions recursively."""
        storage = self.symmetryfunctions.copy()
        for sfset in self.sets:
            storage += sfset.storage

        return storage

    @property
    def sftypes(self) -> Optional[str]:

        sftypes = list(set(sf.tag for sf in self.symmetryfunctions))
        if len(sftypes) == 1:
            return sftypes[0]

        if len(sftypes) > 1:
            return 'mixed'

        return None

    @property
    def elements(self) -> Optional[list[str]]:

        # Store all elements of all symmetryfunctions.
        elements = []
        for symfun in self.symmetryfunctions:
            if symfun.elements is not None:
                elements += symfun.elements

        # Remove duplicates.
        elements = list(set(elements))

        # If the list is empty, return None instead.
        if len(elements) == 0:
            return None

        return elements

    @property
    def cutoffs(self) -> Optional[list[float]]:
        cutoffs = list(set(sf.cutoff for sf in self.symmetryfunctions))

        if len(cutoffs) == 0:
            return None

        return cutoffs

    def append(
        self,
        blob: Union['SymmetryFunctionSet', SymmetryFunction]
    ) -> None:
        """Append a data `blob` to the internal storage."""
        if isinstance(blob, SymmetryFunctionSet):
            self._sets.append(blob)

        elif isinstance(blob, SymmetryFunction):
            self._symmetryfunctions.append(blob)

        else:
            raise Exception(f'{self.__class__.__name__} can only store data of'
                           + 'type SymmetryFunctionSet or SymmetryFunction.')


# def get_radial_coefficients_turn(
#     rturn: float,
#     cutoff: float
# ) -> npt.NDArray[float]:
#     """Calculate coefficients of one radial symfun with turnpoint at `rturn`."""
#     phi = np.pi * rturn / cutoff
#     cosphi: float = np.cos(phi)
#     sinphi: float = np.sin(phi)
# 
#     df1 = 2.0 * (cosphi + 1.0)
#     df2 = 8.0 * df1 * rturn**2
#     df3 = 2.0 * df1 - 4.0 * phi * sinphi
#     sqrtterm: float = np.sqrt(df3**2 + df2 * np.pi**2 / cutoff**2 * cosphi)
#     eta = (df3 + sqrtterm) / df2
# 
#     coefficients: npt.NDArray[float] = np.array([eta, 0.0])
# 
#     return coefficients
# 
# 
# def get_radial_coefficients_half(
#     rturn: float,
#     cutoff: float
# ) -> npt.NDArray[float]:
#     """Calculate coefficients of one radial symfun where f(`rturn`) = 0.5."""
#     phi = np.pi * rturn / cutoff
#     cosphi: float = np.cos(phi)
#     logphi: float = np.log(cosphi + 1.0)
#     eta = logphi / rturn**2
# 
#     coefficients: npt.NDArray[float] = np.array([eta, 0.0])
# 
#     return coefficients
# 
# 
# def calculate_radial_coefficients(
#     sfset: SymmetryFunctionSet,
#     algorithm: str
# ) -> None:
#     """Calculate the coefficients of radial symmetry functions."""
# 
#     cutoff = sfset.cutoffs[0]
#     elements = sfset.elements
#     rmin = sfset.min_distances['-'.join(elements)]
# 
#     dturn: float = 0.5 * cutoff - rmin
#     interval: float = dturn / float(len(sfset) - 1.0)
# 
#     for idx, symfun in enumerate(sfset.symmetryfunctions):
#         rturn: float = 0.5 * cutoff - interval * float(idx)
# 
#         # Equally spaced at G(r) = 0.5.
#         if algorithm == 'half':
#             symfun.coefficients = get_radial_coefficients_half(rturn, cutoff)
# 
#         # Equally spaced turning points.
#         elif algorithm == 'turn':
#             symfun.coefficients = get_radial_coefficients_turn(rturn, cutoff)
#         else:
#             raise Exception('Unknown algorithm.')


def get_element_groups(
    elements: list[str],
    groupsize: int
) -> list[list[str]]:
    """Create doubles or triplets of elements from all `elements`.

    Arguments
    ---------
    elements : list of str
        A list of all the elements from which the groups shall be built.
    groupsize : int
        The desired size of the group.

    Returns
    -------
    groups : list of lists of str
        A list of elements groups.
    """
    # Build pairs of elements.
    if groupsize == 2:
        doubles = list(product(elements, repeat=2))
        groups = [[a, b] for (a, b) in doubles]

    # Build triples of elements.
    elif groupsize == 3:
        pairs = combinations_with_replacement(elements, 2)
        triples = product(pairs, elements)
        groups = [[a, b, c] for (a, b), c in triples]

    return groups



    # 
    # def set_coefficients(
    #     self,
    #     algorithm: str = 'half',
    #     elements: Optional[list[str]] = None
    # ) -> None:
    # 
    #     if self.cutoffs and len(self.cutoffs) != 1:
    #         raise Exception('Can only calculate coefficients for same cutoff radius.')

        # Determine the size of the groups.
        # if symmetryfunction.tag == 'radial':
        #     groupsize = 2
        # elif symmetryfunction.tag == 'angular':
        #     groupsize = 3
        # 
        # if elements is None:
        #     if symmetryfunction.elements is None:
        #         raise Exception('Please either specify elements or symmetryfunction.elements.')
        # 
        #     groups = [symmetryfunction.elements]
        # 
        # elif isinstance(elements, list):
        # 
        #     # If a list of elements was given set all permutations.
        #     # Otherwise, use the user-defined list of permutations.
        #     if all(isinstance(i, str) for i in elements):
        #         groups = get_element_groups(elements, groupsize)
        #     else:
        #         if any(len(i) != groupsize for i in elements):
        #             raise Exception('Symmetry function type and group size do not match.')
        #         groups = elements
        # 
        # for elem_group in groups:
        #     print(elem_group)
        #     # Get the label of this element combination.
        #     # label = '-'.join(elem_group)
        #     #rmin = {label: self.min_distances[label]}
        #     # Create a new symmetry function set.
        #     sfset = SymmetryFunctionSet()
        # 
        # 
        # 
        # 
        # 
        # 
        # if self.sftypes == 'radial':
        #     if self.min_distances is None:
        #         raise Exception('Need minimum distances to set radial coefficients.')
        # 
        # for elem_group in groups:
        #     print(elem_group)
        #     # Get the label of this element combination.
        #     # label = '-'.join(elem_group)
        #     #rmin = {label: self.min_distances[label]}
        #     # Create a new symmetry function set.
        #     sfset = SymmetryFunctionSet()
        # 
        #     # Append symmetry functions to the new set.
        #     for idx in range(size):
        # 
        #         symfun = symmetryfunction.copy()
        #         symfun.elements = elem_group
        #         sfset += symfun
        # 
        # 
        #     calculate_radial_coefficients(self, algorithm)
        # 
        # if self.sftypes == 'angular':
        #     calculate_angular_coefficients(self, algorithm)
        # 
        # else:
        #     raise Exception('Cannot calculate coefficients for mixed sets.')
        # 
        # 

# if __name__ == '__main__':
# 
#     radials = SymmetryFunction(2, 10.0)
#     radialset = 6 * radials
#     radialset.set_coefficients(algorithm='turn')
# 
#     angulars = SymmetryFunction(3, 12.0)
#     angularset = 8 * angulars

    # sfset = radialset + angularset
    # print(sfset)

    # c = SymmetryFunctionSet()
    # c.append(radialset)
    # c.append(angularset)
    # print(c)
    # a.add_symmetryfunctions(size=6, sftype=2, 10.0)

    # print(a)
    # print(a.sets)
    # print(a.cutoffs)
    # print(a.elements)
    # print(a.sftypes)

    #     if size >= 1 and symmetryfunction is not None:
    #         self.add_symmetryfunctions(size, symmetryfunction, elements)
    # 
    # def add_symmetryfunctions(
    #     self,
    #     size: int,
    #     symmetryfunction: SymmetryFunction,
    #     elements: Optional[Union[list[str], list[list[str]]]] = None,
    # ) -> None:
    #     """Add an amount of `size` symmetryfunctions to this set."""
    #     # Determine the size of the groups.
    #     if symmetryfunction.tag == 'radial':
    #         groupsize = 2
    #     elif symmetryfunction.tag == 'angular':
    #         groupsize = 3
    # 
    #     if elements is None:
    #         if symmetryfunction.elements is None:
    #             raise Exception('Please either specify elements or symmetryfunction.elements.')
    # 
    #         groups = [symmetryfunction.elements]
    # 
    #     elif isinstance(elements, list):
    # 
    #         # If a list of elements was given set all permutations.
    #         # Otherwise, use the user-defined list of permutations.
    #         if all(isinstance(i, str) for i in elements):
    #             groups = get_element_groups(elements, groupsize)
    #         else:
    #             if any(len(i) != groupsize for i in elements):
    #                 raise Exception('Symmetry function type and group size do not match.')
    #             groups = elements
    # 
    #     for elem_group in groups:
    #         print(elem_group)
    #         # Get the label of this element combination.
    #         # label = '-'.join(elem_group)
    #         #rmin = {label: self.min_distances[label]}
    #         # Create a new symmetry function set.
    #         sfset = SymmetryFunctionSet()
    # 
    #         # Append symmetry functions to the new set.
    #         for idx in range(size):
    # 
    #             symfun = symmetryfunction.copy()
    #             symfun.elements = elem_group
    #             sfset += symfun
    # 
    #         # Add the set of symmetry functions to this object.
    #         self += sfset
    # 
    #         if isinstance(coefficients, str):
    #             if symmetryfunction.sftype == 2:
    #                 calculate_radial_coefficients(sfset, coefficients)
    # 
    #         #     # elif sftype == 3:
    #         #     #     calculate_angular_coefficients(sfset, coefficients)
    #         # 
    #         # self.append(sfset)    
    
    # a += SymmetryFunction(2, 10.0)

    # b = SymmetryFunctionSet(['C', 'H'])
    # b += SymmetryFunction(3, 12.0, elements=['C', 'H'])
    # 
    # c = SymmetryFunctionSet(['C', 'H'])
    # c += SymmetryFunction(2, 14.0)
    # 
    # a.append(b)
    # 
    # c.append(a)
    # 
    # # c = a + b
    # 
    # print('a', a)
    # print('b', b)
    # print('c', c)
    # print('c', c.sets)
    # print('c', c.symmetryfunctions)
    # print('c', c.storage)
    # 
    # c += SymmetryFunction(3, 12.0, elements=['C', 'H'])
    # 
    # print(c.cutoffs)




    # def add_radil(
    #     self,
    #     sftype: int = 2,
    #     cutoff: float = 10.0,
    #     amount: int = 6,
    #     elements: Optional[list[str]] = None,
    #     algorithm: str = 'half'
    # ) -> None:
    #     """Add a set of radial symmetry functions to this set."""
    # 
    #     # If no elements were provided use all elements of this set.
    #     if elements is None:
    #         elements = self.elements
    # 
    #     # Radial symmetry functions require min_distances to be defined.
    #     if self.min_distances is None:
    #         raise Exception('Please provide minimum pairwise distances.')
    # 
    # 
    #     # Create one set of symmetry functions for each element pair.
    #     for elements_group in get_element_groups(elements, 2):
    #         label = '-'.join(elements_group)
    #         rmin = {label: self.min_distances[label]}
    #         symfunset = SymmetryFunctionSet(elements_group, rmin)
    # 
    #         # Set the symmetry function coefficients. This modifies symfunset.
    #         calc_symfunc_radial(symfunset, cutoff, self.min_distances[label],
    #                             size, algorithm)
    # 
    #         # Append the new set of symmetry functions to this set.
    #         self += symfunset

    # # pylint: disable=R0913
    # def add_radial(
    #     self,
    #     sftype: int = 2,
    #     cutoff: float = 10.0,
    #     size: int = 6,
    #     elements: Optional[list[str]] = None,
    #     algorithm: str = 'half'
    # ) -> None:
    #     """Add a set of radial symmetry functions to this set."""
    # 
    #     # If no elements were provided use all elements of this set.
    #     if elements is None:
    #         elements = self.elements
    # 
    #     # Radial symmetry functions require min_distances to be defined.
    #     if self.min_distances is None:
    #         raise Exception('Please provide minimum pairwise distances.')
    # 
    #     # Create one set of symmetry functions for each element pair.
    #     for elements_group in get_element_groups(elements, 2):
    #         label = '-'.join(elements_group)
    #         rmin = {label: self.min_distances[label]}
    #         symfunset = SymmetryFunctionSet(elements_group, rmin)
    # 
    #         # Set the symmetry function coefficients. This modifies symfunset.
    #         calc_symfunc_radial(symfunset, cutoff, self.min_distances[label],
    #                             size, algorithm)
    # 
    #         # Append the new set of symmetry functions to this set.
    #         self.append(symfunset)

    # # pylint: disable=R0913
    # def add_radialset(
    #     self,
    #     sftype: int = 2,
    #     cutoff: float = 10.0,
    #     size: int = 6,
    #     elements: Optional[list[str]] = None,
    #     algorithm: str = 'half'
    # ) -> None:
    #     """Add a set of radial symmetry functions to this set."""
    # 
    #     # If no elements were provided use all elements of this set.
    #     if elements is None:
    #         elements = self.elements
    # 
    #     # Radial symmetry functions require min_distances to be defined.
    #     if self.min_distances is None:
    #         raise Exception('Please provide minimum pairwise distances.')
    # 
    #     # Create one set of symmetry functions for each element pair.
    #     for elements_group in get_element_groups(elements, 2):
    #         label = '-'.join(elements_group)
    #         rmin = {label: self.min_distances[label]}
    #         symfunset = SymmetryFunctionSet(elements_group, rmin)
    # 
    #         # Set the symmetry function coefficients. This modifies symfunset.
    #         calc_symfunc_radial(symfunset, cutoff, self.min_distances[label],
    #                             size, algorithm)
    # 
    #         # Append the new set of symmetry functions to this set.
    #         self.append(symfunset)


# def get_angular_coefficients_turn(
#     turn: float,
#     lamb: float
# ) -> npt.NDArray[float]:
#     """Calculate coefficients of one radial symfun with turnpoint at `rturn`."""
#     costurn: float = np.cos(turn)
#     sinturn: float = np.sin(turn)
#     rho = 1.0 + lamb * costurn
#     zeta = 1.0 + (costurn / sinturn**2) * rho / lamb
# 
#     coefficients: npt.NDArray[float] = np.array([lamb, zeta, 0.0])
# 
#     return coefficients
# 
# 
# def get_angular_coefficients_half(
#     turn: float,
#     lamb: float
# ) -> npt.NDArray[float]:
#     """Calculate coefficients of one radial symfun with turnpoint at `rturn`."""
#     costurn: float = np.cos(turn)
#     rho = 1.0 + lamb * costurn
#     logrho: float = np.log(rho)
#     zeta: float = -np.log(2) / (logrho - np.log(2))
# 
#     coefficients: npt.NDArray[float] = np.array([lamb, zeta, 0.0])
# 
#     return coefficients
# 
# 
# def calc_symfunc_angular(
#     symfuns: SymmetryFunctionSet,
#     algorithm: str,
#     lambas: list[float] = [1.0, -1.0]
# ) -> SymmetryFunctionSet:
#     """Calculate the coefficients of radial symmetry functions."""
#     lambdas = [1.0, -1.0]
#     interval = 160.0 / len(symfuns)
# 
#     for idx, symfun in enumerate(symfuns):
#         turn: float = (160.0 - interval * idx) / 180.0 * np.pi
# 
#         # Equally spaced at G(r) = 0.5.
#         if algorithm == 'half':
#             symfun.coefficients = get_radial_coefficients_half(turn, lamb)
# 
#         # Equally spaced turning points.
#         elif algorithm == 'turn':
#             symfun.coefficients = get_radial_coefficients_turn(turn, cutoff)
# 
#         elif algorithm == 'literature':
#             symfun.coefficients = get_radial_coefficients_literature(turn,
#                                                                      cutoff)
# 
#     for lamb in lambdas:
#         zeta = np.zeros(n_angular)
#         for i in range(n_angular):
# 
#             # Get the next point of reference.
#             tturn = (160.0 - interval * i) / 180.0 * np.pi
# 
#             # Equally spaced at G(r) = 1.0.
#             if algorithm == 'half':
#                 rho = 1.0 + lamb * np.cos(tturn)
#                 zeta[i] = -np.log(2) / (np.log(rho) - np.log(2))
# 
#             # Equally spaced turning points.
#             elif algorithm == 'turn':
#                 rho = 1.0 + lamb * np.cos(tturn)
#                 zeta[i] = 1 + (np.cos(tturn) / np.sin(tturn)**2) * rho / lamb
# 
#             # Literature turning points.
#             elif algorithm == 'literature':
#                 zeta[i] = 2*(i + 1)
# 
#     return symfuns
# 
# def calc_angular_symfuns(n_angular, algorithm):
#     """Calculate the coefficients of angular symmetry functions."""
#     # Hard-coded literature values for the zeta parameter.
#     zeta_lit = [1.0, 2.0, 4.0, 16.0, 64.0]
#     lambdas = [1.0, -1.0]
# 
#     # Calculate the interval between reference points.
#     interval = 160.0 / n_angular
# 
#     # Calculate the zeta values for each symmetry function.
#     params = []
# 
#     for lamb in lambdas:
#         zeta = np.zeros(n_angular)
#         for i in range(n_angular):
# 
#             # Get the next point of reference.
#             tturn = (160.0 - interval * i) / 180.0 * np.pi
# 
#             # Equally spaced at G(r) = 1.0.
#             if algorithm == 'half':
#                 rho = 1.0 + lamb * np.cos(tturn)
#                 zeta[i] = -np.log(2) / (np.log(rho) - np.log(2))
# 
#             # Equally spaced turning points.
#             elif algorithm == 'turn':
#                 rho = 1.0 + lamb * np.cos(tturn)
#                 zeta[i] = 1 + (np.cos(tturn) / np.sin(tturn)**2) * rho / lamb
# 
#             # Literature turning points.
#             elif algorithm == 'literature':
#                 if n_angular > 5:
#                     raise PropertyNotImplementedError('Literature algorithm '
#                                                       + 'only works with '
#                                                       + 'n_angular <= 4.')
#                 zeta[i] = zeta_lit[i]
# 
#         params.append([lamb, zeta])
# 
#     return params
# 
# 
# 




#pylint: disable=[R0913, R0914]
# def calc_symfun_coefficients(
#     elements,
#     cutoff=12.0,
#     dataset=None,
#     rmins=None,
#     n_radial=6,
#     n_angular=4,
#     algorithm_radial='half',
#     algorithm_angular='literature'
# ) -> list[SymmetryFunction]:
#     """Calculate radial and angular symmetry functions.
# 
#     In RuNNer, the environment around an atom up to a `cutoff` radius is
#     described by many-body symmetry functions. _Radial_ symmetry functions are
#     two-body terms, while _angular_ symmetry functions depend on a group of
#     three atoms. Usually, multiple symmetry functions with different
#     coefficients are employed.
#     This routine offers a convenient way to calculate these symmetry functions.
# 
#     Arguments
#     ---------
#     dataset: list of `Atoms` objects
#         Symmetry functions are element-specific and are tailored to the minimal
#         interatomic distances in a RuNNer training dataset. Therefore, the
# 
#     Reference
#     ---------
#     * Further information on symmetry functions is provided in
#       [the online documentation of RuNNer](https://theochem.gitlab.io/runner).
# 
#     """
#     if dataset is None and rmins is None:
#         raise CalculatorSetupError('Please provide either a dataset or a dicts '
#                                    + 'of minimal distances.')
# 
#     # If unspecified, determine `rmins` based on the provided dataset.
#     if rmins is None:
#         rmins = get_minimum_distances(dataset, elements)
# 
#     if elements is None:
#         raise CalculatorSetupError('No elements specified. Please check the '
#                                    + 'calculation parameters.')
# 
#     symmetryfunctions = []
# 
#     # Calculate radial symmetry functions === two-body terms.
#     for elements_group in get_element_groups(elements, 2):
#         label = '-'.join(elements_group)
#         coefficients = calc_radial_symfuns(cutoff, rmins[label],
#                                            n_radial, algorithm_radial)
# 
#         for coefficient in coefficients:
#             new_symfun = coefficient_to_symfun(label, coefficient, cutoff)
#             symmetryfunctions.append(new_symfun)
# 
#     # Calculate angular symmetry functions === three-body terms.
#     for elements_group in get_element_groups(elements, 3):
#         label = '-'.join(elements_group)
#         coefficients = calc_angular_symfuns(n_angular, algorithm_angular)
# 
#         for lambd, zetas in coefficients:
#             for zeta in zetas:
#                 new_symfun = coefficient_to_symfun(label, (lambd, zeta), cutoff)
#                 symmetryfunctions.append(new_symfun)
# 
#     return symmetryfunctions


# def coefficient_to_symfun(label, coefficient, cutoff):
#     """Calculate this is the docstring."""
#     elements = label.split('-')
# 
#     # Two elements indicates a radial symmetry function.
#     if len(elements) == 2:
#         elem1, elem2 = elements
#         eta = coefficient
#         symmetryfunction = [elem1, 2, elem2, eta, 0.0, cutoff]
# 
#     elif len(elements) == 3:
#         elem1, elem2, elem3 = elements
#         lamb, zeta = coefficient
#         symmetryfunction = [elem1, 3, elem2, elem3, 0.0, lamb, zeta, cutoff]
# 
#     return symmetryfunction





# def calc_angular_symfuns(n_angular, algorithm):
#     """Calculate the coefficients of angular symmetry functions."""
#     # Hard-coded literature values for the zeta parameter.
#     zeta_lit = [1.0, 2.0, 4.0, 16.0, 64.0]
#     lambdas = [1.0, -1.0]
# 
#     # Calculate the interval between reference points.
#     interval = 160.0 / n_angular
# 
#     # Calculate the zeta values for each symmetry function.
#     params = []
# 
#     for lamb in lambdas:
#         zeta = np.zeros(n_angular)
#         for i in range(n_angular):
# 
#             # Get the next point of reference.
#             tturn = (160.0 - interval * i) / 180.0 * np.pi
# 
#             # Equally spaced at G(r) = 1.0.
#             if algorithm == 'half':
#                 rho = 1.0 + lamb * np.cos(tturn)
#                 zeta[i] = -np.log(2) / (np.log(rho) - np.log(2))
# 
#             # Equally spaced turning points.
#             elif algorithm == 'turn':
#                 rho = 1.0 + lamb * np.cos(tturn)
#                 zeta[i] = 1 + (np.cos(tturn) / np.sin(tturn)**2) * rho / lamb
# 
#             # Literature turning points.
#             elif algorithm == 'literature':
#                 if n_angular > 5:
#                     raise PropertyNotImplementedError('Literature algorithm '
#                                                       + 'only works with '
#                                                       + 'n_angular <= 4.')
#                 zeta[i] = zeta_lit[i]
# 
#         params.append([lamb, zeta])
# 
#     return params
