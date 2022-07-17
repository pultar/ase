#!/usr/bin/env python3
# encoding: utf-8

"""Define regression tests for RuNNer."""

from typing import List

import pytest

from ase.atoms import Atoms
from ase.calculators.runner.runner import Runner
from ase.io.runner.symmetryfunctions import generate_symmetryfunctions


class TestFunctional:
    """Define functional tests for the runner calculator.

    These tests pass if they complete without an error.
    """

    seed = 42

    @pytest.mark.calculator('runner')
    @pytest.mark.parametrize('dataset', ['dataset_h2o'])
    def test_modes(self, dataset: List[Atoms], factory, request) -> None:
        """Run RuNNer Modes 1 through 3 for a given test system."""
        # Load the test dataset.
        dataset = request.getfixturevalue(dataset)

        # Create a calculator object using the ASE interface and run the job.
        runnercalc = Runner(dataset=dataset)

        # Define symmetry functions.
        radials = generate_symmetryfunctions(dataset, sftype=2,
                                             algorithm='half')
        angulars = generate_symmetryfunctions(dataset, sftype=3,
                                              algorithm='literature')

        runnercalc.symmetryfunctions += radials
        runnercalc.symmetryfunctions += angulars

        # Set a defined seed for reproducibility.
        runnercalc.set(random_seed=self.seed)

        # Run Mode 1.
        runnercalc.run(mode=1)

        # Run Mode 2.
        runnercalc.set(epochs=3)
        runnercalc.set(use_short_forces=False)
        runnercalc.run(mode=2)

        # Run Mode 3.
        runnercalc.run(mode=3)

    @pytest.mark.calculator('runner')
    @pytest.mark.parametrize('dataset', ['dataset_h2o'])
    def test_restart(self, dataset: List[Atoms], factory, request) -> None:
        """Test that the RuNNer calculator can restart without any changes."""
        # Load the test dataset.
        dataset = request.getfixturevalue(dataset)

        # Create a calculator object using the ASE interface and run the job.
        runnercalc = Runner(dataset=dataset)

        # Define symmetry functions.
        radials = generate_symmetryfunctions(dataset, sftype=2,
                                             algorithm='half')
        angulars = generate_symmetryfunctions(dataset, sftype=3,
                                              algorithm='literature')

        runnercalc.symmetryfunctions += radials
        runnercalc.symmetryfunctions += angulars

        # Set a defined seed for reproducibility.
        runnercalc.set(random_seed=self.seed)

        # Run Mode 1.
        runnercalc.run(mode=1)

        runnercalc_restarted = Runner(restart='mode1/mode1')

        assert len(runnercalc.results) == len(runnercalc_restarted.results)
        assert all(i is not None for i in runnercalc.results.values())
