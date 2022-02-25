"""This module defines an ASE Calculator implementing the spin-spacing avareging (SSA) method

Spin-space averaging method published in F. Körmann, A. Dick,
B. Grabowski, T. Hickel, and J. Neugebauer, Phys. Rev. B 85, 125104
(2012).

Compute calculator results by repeatedly running the underlying calculator, each
time with a different initial_magnetic_moments, and averaging the output quantities in the
results dict.

Written by:

    Noam Bernstein noam.bernstein@nrl.navy.mil
"""

from pathlib import Path

from ase.calculators.calculator import all_changes

def SSA(base_calc, SSA_weights='uniform', *args, **kwargs):
    """Creates SSA Calculator object

    Parameters:

    base_calc: ASE Calculator class
        class (not instance) of base calculator that will be used multiple times, one for each spin order

    SSA_weights: str, default "uniform"
        method for averaging results.  Supported values: ["uniform"]

    args, kwargs:
        positional and keyword arguments for base_calc constructor

    Returns:

    instance of SSA class object derived from base_calc with the calculate() method which does
        spin-space averaging
    """

    class SSA_class(base_calc):

        def __init__(self, SSA_weights='uniform', *args, **kwargs):
            self.SSA_weights = SSA_weights
            super().__init__(*args, **kwargs)

        def _get_name(self) -> str:  # child class overriding (as per comment in calculator.py)
            return base_calc.__name__.lower()

        def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
            use_atoms = atoms if atoms is not None else self.atoms

            if 'SSA_initial_magnetic_moments' in use_atoms.arrays:
                SSA_initial_magnetic_moments = use_atoms.arrays['SSA_initial_magnetic_moments'].T
            else:
                SSA_initial_magnetic_moments = [use_atoms.get_initial_magnetic_moments()]

            SSA_N = len(SSA_initial_magnetic_moments)
            try:
                orig_directory = self.directory
            except AttributeError:
                assert SSA_N == 1, f'Calculator has no `self.directory`, so only single spin configuration is supported (got {SSA_N})'
                orig_directory = None

            results = {}
            for SSA_i, initial_magnetic_moments in enumerate(SSA_initial_magnetic_moments):
                # each SSA instance in its own directory
                if orig_directory is not None:
                    self.directory = Path(orig_directory) / f'run_SSA_instance_{SSA_i}'

                # set moments for this one
                use_atoms.set_initial_magnetic_moments(initial_magnetic_moments)
                # do calculation
                super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

                if self.SSA_weights == 'uniform':
                    SSA_weight = 1.0 / SSA_N
                else:
                    raise ValueError(f'Unsupposed SSA_weights {self.SSA_weights}')

                # copy back, accumulating
                for k, v in self.results.items():
                    if k in results:
                        results[k] += SSA_weight * v
                    else:
                        results[k] = SSA_weight * v

            self.results.update(results)

            if orig_directory is not None:
                self.directory = orig_directory

    return SSA_class(SSA_weights, *args, **kwargs)
