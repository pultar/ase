#!/usr/bin/env python3
# encoding: utf-8

"""Define regression tests for RuNNer."""

import os

import pytest

from ase.calculators.runner.runner import Runner
import ase.io.runner.runner as io
from ase.io import read


@pytest.fixture(name='system', params=['resources'])
def fixture_system(request):
    """Configure a test system with fixed dataset and RuNNer options."""
    # Get the path where the test files reside.
    path = request.param
    test_dir, _ = os.path.splitext(request.module.__file__)
    res_directory = os.path.join(test_dir, path)

    print(os.listdir(test_dir))
    print(os.listdir(res_directory))

    # Read in the dataset and options from the example.
    dataset = read(os.path.join(res_directory, 'input.data'), index=':')
    options = io.read_runnerconfig(os.path.join(res_directory, 'input.nn'))

    return dataset, options


class TestFunctional:
    """Define functional tests for the runner calculator.

    These tests pass if they complete without an error.
    """

    seed = 42

    @pytest.mark.calculator('runner')
    def test_modes(self, system, factory):
        """Run RuNNer Mode 1 for a given test system."""
        # Get the dataset and parameters.
        dataset, options = system

        # Create a calculator object using the ASE interface and run the job.
        runnercalc = Runner(dataset=dataset, **options)
        runnercalc.set(random_seed=self.seed)
        runnercalc.run(mode=1)

        runnercalc = Runner(restart='mode1/mode1')
        runnercalc.set(epochs=2)
        runnercalc.set(use_short_forces=False)
        runnercalc.run(mode=2)

        runnercalc = Runner(restart='mode2/mode2')
        runnercalc.run(mode=3)
