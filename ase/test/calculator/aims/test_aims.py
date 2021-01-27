"""Test FHI-aims ASE calculator class implemenation.

The class if found here: ase/calculators/aims.py."""

import ase.calculators.aims


def setup_test_aims_calc(outfilename):
    """Setup an aims calc object for tests."""
    run_command = 'mpiexec -np 4 aims.x >' + outfilename
    return ase.calculators.aims.Aims(
        run_command=run_command)


def test_read_number_of_bands():
    """Test reading number of KS states/nbands.

    We test using a aims.out file for Ca4Ta4 that uses
    the light/minimal settings. This file has some strange
    corner case behaviour. You can see this in the section
    labelled 'Structure dependent array size parameters'
    where the 'Maximum number of basis functions' is 180
    and 'Number of Kohn-Sham states (occupied + empty)'
    is larger at 206. Normally the # of KS states should
    always be smaller than the number of basis functions.
    Later on in the aims.out file we see the number of KS
    states is reduced to 180. We want to check that this
    method returns the number of bands as being 180 and not
    206 for this aims.out file.
    """
    test_file = (
        'ase/test/calculator/test_data/'
        'aims/read_number_of_bands/aims.out')
    ase_io_obj = setup_test_aims_calc(
        outfilename=test_file)
    nbands = ase_io_obj.read_number_of_bands()
    assert nbands == 180
