"""Test FHI-aims ASE calculator class implemenation.

The class is found here: ase/calculators/aims.py."""

import ase.calculators.aims


def write_partial_aims_outfile(outfile_path: str, file_content: str):
    """Create a text file with the file_content string written to it.

    New text file named outfile_path is created. file_content is written
    to the text file.

    Args:
        outfile_path: Path to output file.
        file_content: What should be written to the output file.
    """
    with open(outfile_path, 'w') as outfile:
        outfile.write(file_content)


def setup_test_aims_calc(outfile_path: str):
    """Setup an aims calcator object for tests."""
    run_command = 'mpiexec -np 4 aims.x >' + outfile_path
    # Return an ase Aims calculator object
    return ase.calculators.aims.Aims(
        run_command=run_command)


class TestAimsCalculator():
    """Test class for testing aims calcutor."""
    def test_read_number_of_bands(self, tmpdir):
        """Test reading number of KS states/nbands.

        We test using a string derived from a real aims.out file for Ca4Ta4 that uses
        the light/minimal settings. This string contains has some strange corner case behaviour.
        You can see this in the section labelled `Structure dependent array size parameters` where
        the 'Maximum number of basis functions' is 180 and `Number of Kohn-Sham states
        (occupied + empty)` is larger at 206. Normally the # of KS states should always be smaller
        than the number of basis functions. Later on in the aims.out file we see the number of KS
        states is reduced to 180. We want to check that this method returns the number of bands as
        being 180 and not 206 for this string taken from a real aims.out output file.

        Args:
            tmpdir: This is used by pytest to tell it to create a temporary directory. We will
                write a string to a text file in this tempdir for this unit test. We don't need
                the file afterwards.
        """
        nbands_str = """
            Number of empty states per atom not set in control.in - providing a guess from actual geometry.
            | Total number of empty states used during s.c.f. cycle:       48
            If you use a very high smearing, use empty_states (per atom!) in control.in to increase this value.

            Structure-dependent array size parameters: 
            | Maximum number of distinct radial functions  :       17
            | Maximum number of basis functions            :      180
            | Number of Kohn-Sham states (occupied + empty):      206
            ------------------------------------------------------------
            ...
            ...
            Reducing total number of  Kohn-Sham states to      180."""
        # Create a temporary directory using pytest's tmpdir and a temporary file.
        # We cast the output of the command to a string which we can pass to other functions.
        outfile_path = str(tmpdir.mkdir("sub").join("temp_aims_output.out"))
        # Create a text file that contains nbands_str so we can test whether
        # relevant information in the string is parsed correctly. This should
        # provide a good test for the real output file since this string
        # is taken from a real FHI-aims output file.
        write_partial_aims_outfile(
            outfile_path=outfile_path,
            file_content=nbands_str)
        ase_io_obj = setup_test_aims_calc(outfile_path=outfile_path)
        nbands = ase_io_obj.read_number_of_bands()
        assert nbands == 180
