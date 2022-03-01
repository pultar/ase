"""Implementation of classes for storing RuNNer parameters and results.

This module provides custom classes for storing the different types of data
produced and/or read by RuNNer.

Provides
--------
get_element_groups : utility function
    Create a list of element pairs and triples from a list of chemical symbols.
get_minimum_distances : utility function
    Find the minimum distance for each pair of elements in a list of images.

Reference
---------
- [The online documentation of RuNNer](https://theochem.gitlab.io/runner)

Contributors
------------
- Author: [Alexander Knoll](mailto:alexander.knoll@chemie.uni-goettingen.de)

"""

from typing import Optional, Dict, List

from itertools import combinations_with_replacement, product

import numpy as np

from ase.geometry import get_distances
from ase.atoms import Atoms

from ase.calculators.runner.runner import get_elements
from .storageclasses import SymmetryFunction, SymmetryFunctionSet


def get_element_groups(
    elements: List[str],
    groupsize: int
) -> List[List[str]]:
    """Create doubles or triplets of elements from all `elements`.

    Arguments
    ---------
    elements : list of str
        A list of all the elements from which the groups shall be built.
    groupsize : int
        The desired size of the group.

    Returns
    -------
    groups : list[lists[str]]
        A list of elements groups.
    """
    # Build pairs of elements.
    if groupsize == 2:
        doubles = list(product(elements, repeat=2))
        groups = [[a, b] for (a, b) in doubles]

    # Build triples of elements.
    elif groupsize == 3:
        pairs = combinations_with_replacement(elements, 2)
        triples = product(elements, pairs)
        groups = [[a, b, c] for a, (b, c) in triples]

    return groups


def get_minimum_distances(
    dataset: List[Atoms],
    elements: List[str]
) -> Dict[str, float]:
    """Calculate min. distance between all `elements` pairs in `dataset`.

    Parameters
    ----------
    dataset : List[Atoms]
        The minimum distances will be returned for each element pair across all
        images in `dataset`.
    elements : List[str]
        The list of elements from which a list of element pairs will be built.

    Returns
    -------
    minimum_distances: Dict[str, float]
        A dictionary where the keys are strings of the format 'C-H' and the
        values are the minimum distances of the respective element pair.

    """
    minimum_distances: Dict[str, float] = {}
    for elem1, elem2 in get_element_groups(elements, 2):
        for structure in dataset:

            elems = structure.get_chemical_symbols()

            # All positions of one element.
            pos1 = structure.positions[np.array(elems) == elem1]
            pos2 = structure.positions[np.array(elems) == elem2]

            distmatrix = get_distances(pos1, pos2)[1]

            # Remove same atom interaction.
            flat = distmatrix.flatten()
            flat = flat[flat > 0.0]

            dmin: float = min(flat)
            label = '-'.join([elem1, elem2])

            if label not in minimum_distances:
                minimum_distances[label] = dmin

            # Overwrite the currently saved minimum distances if a smaller one
            # has been found.
            if minimum_distances[label] > dmin:
                minimum_distances[label] = dmin

    return minimum_distances


# This wrapper needs all the possible symmetry function arguments.
# pylint: disable=R0913
def generate_symmetryfunctions(
    dataset: List[Atoms],
    sftype: int = 2,
    cutoff: float = 10.0,
    amount: int = 6,
    algorithm: str = 'half',
    elements: Optional[List[str]] = None,
    min_distances: Optional[Dict[str, float]] = None,
    lambda_angular: Optional[List[float]] = None
) -> SymmetryFunctionSet:
    """Based on a dataset, generate a set of radial symmetry functions."""
    # If no elements were provided use all elements of this set.
    if elements is None:
        elements = get_elements(dataset)

    # Generate the parent symmetry function set.
    parent_symfunset = SymmetryFunctionSet()

    # Generate radial symmetry functions.
    if sftype == 2:

        # For radial symmetry functions, minimum element distances are required.
        if min_distances is None:
            min_distances = get_minimum_distances(dataset, elements)

        # Create one set of symmetry functions for each element pair.
        for element_group in get_element_groups(elements, 2):
            # Get label and save the min_distances for this element pair.
            label = '-'.join(element_group)
            rmin = {label: min_distances[label]}

            # Add `amount` symmetry functions to a fresh symmetry function set.
            element_symfunset = SymmetryFunctionSet(min_distances=rmin)
            for _ in range(amount):
                element_symfunset += SymmetryFunction(sftype=2, cutoff=cutoff,
                                                      elements=element_group)

            # Set the symmetry function coefficients. This modifies symfunset.
            generate_symfun_radial(element_symfunset, cutoff, algorithm,
                                   elements=element_group)

            parent_symfunset.append(element_symfunset)

    # Generate angular symmetry functions.
    elif sftype == 3:
        # Set lambda coefficients, if unprovided.
        if lambda_angular is None:
            lambda_angular = [-1.0, +1.0]

        # Create one set of symmetry functions for each element triplet.
        for element_group in get_element_groups(elements, 3):
            for lamb in lambda_angular:

                # Add `amount` symmetry functions to a fresh set.
                element_symfunset = SymmetryFunctionSet()
                for _ in range(amount):
                    element_symfunset += SymmetryFunction(
                        sftype=3,
                        cutoff=cutoff,
                        elements=element_group
                    )

                # Set the symmetry function coefficients. This modifies
                # symfunset.
                generate_symfun_angular(element_symfunset, algorithm, lamb)

                parent_symfunset.append(element_symfunset)

    else:
        raise NotImplementedError('Cannot generate symmetry functions for '
                                  + '`sftype`s other than 2 or 3.')

    return parent_symfunset


def get_radial_coefficients_turn(
    rturn: float,
    cutoff: float
) -> List[float]:
    """Calculate coefficients of one radial symfun with turnpoint at `rturn`."""
    phi = np.pi * rturn / cutoff
    cosphi: float = np.cos(phi)
    sinphi: float = np.sin(phi)

    df1 = 2.0 * (cosphi + 1.0)
    df2 = 8.0 * df1 * rturn**2
    df3 = 2.0 * df1 - 4.0 * phi * sinphi
    sqrtterm: float = np.sqrt(df3**2 + df2 * np.pi**2 / cutoff**2 * cosphi)
    eta = (df3 + sqrtterm) / df2

    return [eta, 0.0]


def get_radial_coefficients_half(
    rturn: float,
    cutoff: float
) -> List[float]:
    """Calculate coefficients of one radial symfun where f(`rturn`) = 0.5."""
    phi = np.pi * rturn / cutoff
    cosphi: float = np.cos(phi)
    logphi: float = np.log(cosphi + 1.0)
    eta = logphi / rturn**2

    return [eta, 0.0]


def generate_symfun_radial(
    sfset: SymmetryFunctionSet,
    cutoff: float,
    algorithm: str,
    elements: List[str]
) -> None:
    """Calculate the coefficients of radial symmetry functions."""
    if sfset.min_distances is not None:
        rmin = sfset.min_distances['-'.join(elements)]
    else:
        rmin = 1.0

    dturn: float = 0.5 * cutoff - rmin
    interval: float = dturn / float(len(sfset) - 1.0)

    for idx, symfun in enumerate(sfset.symmetryfunctions):
        rturn: float = 0.5 * cutoff - interval * float(idx)

        # Equally spaced at G(r) = 0.5.
        if algorithm == 'half':
            symfun.coefficients = get_radial_coefficients_half(rturn, cutoff)

        # Equally spaced turning points.
        elif algorithm == 'turn':
            symfun.coefficients = get_radial_coefficients_turn(rturn, cutoff)

        else:
            raise NotImplementedError(f"Unknown algorithm '{algorithm}'.")


def get_angular_coefficients_turn(
    turn: float,
    lamb: float
) -> List[float]:
    """Calculate coefficients of one radial symfun with turnpoint at `rturn`."""
    costurn: float = np.cos(turn)
    sinturn: float = np.sin(turn)
    rho = 1.0 + lamb * costurn
    zeta = 1.0 + (costurn / sinturn**2) * rho / lamb

    return [0.0, lamb, zeta]


def get_angular_coefficients_half(
    turn: float,
    lamb: float
) -> List[float]:
    """Calculate coefficients of one radial symfun with turnpoint at `rturn`."""
    costurn: float = np.cos(turn)
    rho = 1.0 + lamb * costurn
    logrho: float = np.log(rho)
    zeta: float = -np.log(2) / (logrho - np.log(2))

    return [0.0, lamb, zeta]


def generate_symfun_angular(
    sfset: SymmetryFunctionSet,
    algorithm: str,
    lambda_angular: float
) -> None:
    """Calculate the coefficients of angular symmetry functions."""
    # Calculate the angular range that has to be covered.
    interval = 160.0 / len(sfset)

    for idx, symfun in enumerate(sfset.symmetryfunctions):
        turn: float = (160.0 - interval * idx) / 180.0 * np.pi

        # Equally spaced at G(r) = 0.5.
        if algorithm == 'half':
            symfun.coefficients = get_angular_coefficients_half(turn,
                                                                lambda_angular)

        # Equally spaced turning points.
        elif algorithm == 'turn':
            symfun.coefficients = get_angular_coefficients_turn(turn,
                                                                lambda_angular)

        # Library of literature values.
        elif algorithm == 'literature':
            symfun.coefficients = [0.0, lambda_angular, 2**idx]

        else:
            raise NotImplementedError(f"Unknown algorithm '{algorithm}'.")
