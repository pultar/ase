"""Extension of the SinglePointCalculator class tailored for usage with RuNNer.

The RuNNer Neural Network Energy Representation is a framework for the
construction of high-dimensional neural network potentials developed in the
group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.

Contains
--------

RunnerSinglePointCalculator : SinglePointCalculator
    Extension of the native ASE SinglePointCalculator class.

Reference
---------
* [The online documentation of RuNNer](https://theochem.gitlab.io/runner)

Contributors
------------
* Author: [Alexander Knoll](mailto:alexander.knoll@chemie.uni-goettingen.de)

"""

from typing import Optional

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


class RunnerSinglePointCalculator(SinglePointCalculator):
    """Special calculator for a single configuration, tailored to RuNNer data.

    In addition to the usual properties stored in an ASE SinglePointCalculator
    RuNNer needs the total charge of a structure as separate information.
    Therefore, the `SinglePointCalculator` is extended at this point.

    """

    def __init__(self, atoms: Atoms, **results) -> None:
        """Save energy, forces, ..., and charge for one atomic configuration."""
        # Remove the total charge from the results dictionary.
        totalcharge: Optional[float] = results.pop('totalcharge', None)

        # Initialize the parent class which will handle everything but the
        # total charge.
        SinglePointCalculator.__init__(self, atoms, **results)

        # Store the total charge as part of the results.
        if totalcharge is not None:
            self.results['totalcharge'] = totalcharge
