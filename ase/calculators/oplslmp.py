from pathlib import Path
from ase.io.opls import OPLSff
from ase.calculators.lammpsrun import LAMMPS


class OPLSlmp(LAMMPS):
    implemented_properties = ['energy', 'energies', 'forces']

    def __init__(self, parfile, **kwargs):
        """
        parfile: string
           OPLS parameter file name
        """
        self.oplsff = OPLSff(open(parfile))
        kwargs['tmp_dir'] = 'oplslmp'
        kwargs['keep_tmp_files'] = True

        super().__init__(**kwargs)

        # change defaults to be suitable with OPLS
        self.parameters.update(dict(
            atom_style='full',
        ))
        
    def _write_lammps_data(self, label, tempdir):
        self.parameters.specorder = [
            self.atoms.split_symbol(typ)[0]
            for typ in self.atoms.arrays['types']]

        prefix = str(Path(tempdir) / label)
        self.parameters.update({
            'interactions_file': prefix + '_opls'})
        return self.oplsff.write_lammps(self.atoms, prefix=prefix)
