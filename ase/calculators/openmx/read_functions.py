from __future__ import print_function
from matplotlib import pyplot
from ase.units import Ha, Bohr
from numpy import array, sqrt
from ase.calculators.openmx.import_functions import read_nth_to_last_value
from ase.calculators.openmx.import_functions import remove_infinities


class ReadFunction:
    options = ['NormRD', 'magmom', 'energy', 'magmoms', 'electrons', 'mixing',
               'forces', 'force', 'wannier']

    def __init__(self, function_list=None, spin_polarization='off'):
        self.scf = 0
        self.md = 0
        self.spin_polarization = spin_polarization
        self.atom_mulp_index = -1
        if function_list is None:
            function_list = self.options
        self.function_list = function_list
        self.force = []
        self.forces = []
        self.md_electrons = []
        self.electrons = []
        self.md_magmoms = []
        self.magmoms = []
        self.md_magmom = []
        self.magmom = []
        self.mixing = []
        self.norm_rd = []
        self.energy = []
        self.omega_i = []
        self.total_omega = []
        self.delta_omega_i = []
        self.delta_total_omega = []
        self.wannier_centres = []
        self.spreads = []

    def set_spin_polarization(self, spin_polarization):
        self.spin_polarization = spin_polarization

    def __call__(self, line):
        if ' MD= ' in line:
            if ' SCF= ' in line:
                new_md = int(read_nth_to_last_value(line, 4))
                self.scf = int(read_nth_to_last_value(line, 2))
                if new_md != self.md:
                    if 'NormRD' in self.function_list:
                        self.norm_rd = []
                    if 'magmom' in self.function_list:
                        if self.md:
                            self.md_magmom.append(self.magmom[-1])
                        self.magmom = []
                    if 'energy' in self.function_list:
                        self.energy = []
                    if 'magmoms' in self.function_list:
                        if self.md:
                            for i in range(len(self.magmoms)):
                                if i >= len(self.md_magmoms):
                                    self.md_magmoms.append([])
                                self.md_magmoms[i].append(self.magmoms[i][-1])
                        self.magmoms = []
                    if 'electrons' in self.function_list:
                        if self.md:
                            for i in range(len(self.electrons)):
                                if i >= len(self.md_electrons):
                                    self.md_electrons.append([])
                                self.md_electrons[i].append(
                                                    self.electrons[i][-1])
                        self.electrons = []
                    if 'mixing' in self.function_list:
                        self.mixing = []
                    self.md = new_md
                    return 2

        if 'forces' in self.function_list:
            if 'Fxyz' in line:
                atom_index = int(read_nth_to_last_value(
                                 line, 9).split(',')[0]) - 1
                Fx = float(read_nth_to_last_value(line, 3))
                Fy = float(read_nth_to_last_value(line, 2))
                Fz = float(read_nth_to_last_value(line, 1))
                force_magnitude = sqrt(Fx * Fx + Fy * Fy + Fz * Fz)
                if atom_index >= len(self.forces):
                    self.forces.append([])
                self.forces[atom_index].append(force_magnitude)

        if 'force' in self.function_list:
            if '|Maximum force|read_function' in line:
                self.force.append(float(
                                  read_nth_to_last_value(
                                       line).split('=')[1]) * Ha / Bohr)
        if ' MulP ' in line:
            self.atom_mulp_index += 1
            if 'electrons' in self.function_list:
                if self.atom_mulp_index >= len(self.electrons):
                    self.electrons.append([])
                n = 1
                if self.spin_polarization == 'on':
                    n = 3
                if self.spin_polarization == 'nc':
                    n = 5
                self.electrons[self.atom_mulp_index].append(
                               float(read_nth_to_last_value(line, n)))
            if 'magmoms' in self.function_list and \
                    self.spin_polarization != 'off':
                if self.atom_mulp_index >= len(self.magmoms):
                    self.magmoms.append([])
                if self.spin_polarization == 'on':
                    self.magmoms[self.atom_mulp_index].append(
                                 float(read_nth_to_last_value(line, 1)))
                if self.spin_polarization == 'nc':
                    self.magmoms[self.atom_mulp_index].append(
                                 map(float, (read_nth_to_last_value(line, 3),
                                             read_nth_to_last_value(line, 2),
                                             read_nth_to_last_value(line, 1))))
        if ' MulP:' in line:
            self.atom_mulp_index = -1
        if 'NormRD' in line:
            if 'NormRD' in self.function_list:
                self.norm_rd.append(float(read_nth_to_last_value(line, 4)))
            return 1
        if 'magmom' in self.function_list:
            if 'Total Spin Moment' in line:
                if self.spin_polarization == 'on':
                    self.magmom.append(float(read_nth_to_last_value(line)))
                if self.spin_polarization == 'nc':
                    self.magmom.append(
                        map(float, array(read_nth_to_last_value(line, 3),
                                         read_nth_to_last_value(line, 2),
                                         read_nth_to_last_value(line, 1))))
        if 'energy' in self.function_list:
            if 'dUele' in line:
                self.energy.append(float(read_nth_to_last_value(line)) * Ha)
        if 'mixing' in self.function_list:
            if 'Mixing_weight' in line:
                self.mixing.append(float(read_nth_to_last_value(line)))
        if 'wannier' in self.function_list:
            if 'Center of Wannier Function' in line:
                self.spreads.append([])
                self.wannier_centres.append([])
            if '--->CENT' in line and 'WF' in line:
                self.spreads[-1].append(float(read_nth_to_last_value(line, 2)))
                line = line.replace(',', ' ')
                line = line.replace(')', ' ')
                line = line.replace('(', ' ')
                x = float(read_nth_to_last_value(line, 6))
                y = float(read_nth_to_last_value(line, 5))
                z = float(read_nth_to_last_value(line, 4))
                self.wannier_centres[-1].append((x, y, z))
            if '---> DISE' in line and '(Angs^2)' not in line:

                self.omega_i.append(
                    float(read_nth_to_last_value(line.replace('|', ''), 4)))
                self.delta_omega_i.append(
                    float(read_nth_to_last_value(line.replace('|', ''), 3)))
                if len(self.spreads) > 1:
                    assert len(self.spreads[-1]) == len(self.spreads[-2])
                return 3
            if 'Total_Omega=' in line:
                self.total_omega.append(
                    float(read_nth_to_last_value(line).split('=')[1]))
                if len(self.spreads) > 1:
                    assert len(self.spreads[-1]) == len(self.spreads[-2])
                return 3
            if '---> CONV' in line and 'Angs^2' not in line and '|' in line:
                self.delta_total_omega.append(
                    float(read_nth_to_last_value(line.replace('|', ''), 3)))
                if len(self.spreads) > 1:
                    assert len(self.spreads[-1]) == len(self.spreads[-2])
                return 3
        return 0

    def read_from_file(self, filename):
        with open(filename, 'r') as f:
            line = f.readline()
            while line != '':
                self(line)
                line = f.readline()

    def plot(self, title='', save_to_file=False):
        if 'NormRD' in self.function_list:
            pyplot.plot(range(2, 1 + len(self.norm_rd)),
                        remove_infinities(self.norm_rd[1:]))
            pyplot.xlabel('SCF step')
            pyplot.ylabel('Norm of Residual Density in reciprocal space')
            pyplot.suptitle(title)
            pyplot.semilogy()
            if save_to_file:
                pyplot.savefig(title + 'norm_rd.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
        if 'energy' in self.function_list:
            pyplot.plot(range(2, 1 + len(self.energy)),
                        remove_infinities(self.energy[1:]))
            pyplot.xlabel('SCF step')
            pyplot.ylabel('Energy change from last iteration (eV)')
            pyplot.semilogy()
            pyplot.suptitle(title)
            if save_to_file:
                pyplot.savefig(title + 'energy.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
        if 'magmom' in self.function_list and self.spin_polarization != 'off':
            pyplot.plot(range(1, 1 + len(self.magmom)),
                        remove_infinities(self.magmom))
            pyplot.xlabel('SCF step')
            pyplot.ylabel('Magnetic moment (Bohr Magneton)')
            pyplot.suptitle(title)
            if save_to_file:
                pyplot.savefig(title + 'magmom.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
        if 'magmoms' in self.function_list and self.spin_polarization != 'off':
            for atom_magmom in self.magmoms:
                if self.spin_polarization == 'on':
                    pyplot.plot(range(1, 1 + len(atom_magmom)),
                                remove_infinities(atom_magmom))
            pyplot.ylabel('Magnetic Moment for Each Atom (Bohr Magneton)')
            pyplot.xlabel('SCF step')
            pyplot.suptiread_functiontle(title)
            if save_to_file:
                pyplot.savefig(title + 'magmoms.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
        if 'electrons' in self.function_list:
            for atom_electrons in self.electrons:
                pyplot.plot(range(1, 1 + len(atom_electrons)),
                            remove_infinities(atom_electrons))
            pyplot.ylabel('Mulliken Population for Each Atom')
            pyplot.xlabel('SCF step')
            pyplot.suptitle(title)
            if save_to_file:
                pyplot.savefig(title + 'electrons.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
        if 'mixing' in self.function_list:
            pyplot.plot(range(1, 1 + len(self.mixing)),
                        remove_infinities(self.mixing))
            pyplot.ylabel('Mixing Weight')
            pyplot.xlabel('SCF step')
            pyplot.suptitle(title)
            if save_to_file:
                pyplot.savefig(title + 'mixing.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()

    def plot_md(self, title='', save_to_file=False):
        if 'magmom' in self.function_list and self.spin_polarization != 'off':
            pyplot.plot(range(1, 1 + len(self.md_magmom)),
                        remove_infinities(self.md_magmom))
            pyplot.xlabel('MD step')
            pyplot.ylabel('Magnetic moment (Bohr Magneton)')
            pyplot.suptitle(title)
            if save_to_file:
                pyplot.savefig(title + 'md_magmom.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
        if 'force' in self.function_list:
            pyplot.plot(range(1, len(self.force)),
                        remove_infinities(self.force))
            pyplot.xlabel('MD step')
            pyplot.ylabel('Maximum Force (eV / Ang)')
            pyplot.suptitle(title)
            if save_to_file:
                pyplot.savefig(title + 'force.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
        if 'magmoms' in self.function_list and self.spin_polarization != 'off':
            for atom_magmom in self.md_magmoms:
                if self.spin_polarization == 'on':
                    pyplot.plot(range(1, len(atom_magmom)),
                                remove_infinities(atom_magmom))
            pyplot.ylabel('Magnetic Moment for Each Atom (Bohr Magneton)')
            pyplot.xlabel('MD step')
            pyplot.suptitle(title)
            if save_to_file:
                pyplot.savefig(title + 'md_magmoms.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
        if 'forces' in self.function_list:
            for atom_force in self.forces:
                pyplot.plot(range(1, len(atom_force)),
                            remove_infinities(atom_force))
            pyplot.ylabel('Force on Each Atom (eV / Ang)')
            pyplot.xlabel('MD step')
            pyplot.suptitle(title)
            if save_to_file:
                pyplot.savefig(title + 'forces.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
        if 'electrons' in self.function_list:
            i = 0
            for atom_electrons in self.md_electrons:
                pyplot.plot(range(1, len(atom_electrons)),
                            remove_infinities(atom_electrons), label=str(i))
                i += 1
            pyplot.ylabel('Mulliken Population for Each Atom')
            pyplot.xlabel('MD step')
            pyplot.suptitle(title)
            if save_to_file:
                pyplot.savefig(title + 'md_electrons.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()

    def plot_wannier(self, title='', save_to_file=False):
        if 'wannier' in self.function_list:
            pyplot.plot(range(1, 1 + len(self.omega_i)),
                        remove_infinities(self.omega_i))
            pyplot.xlabel('Iteration step')
            pyplot.ylabel(r'${\Omega}_I / \mathrm{\AA}^2$')
            pyplot.suptitle(title)
            pyplot.ylim(ymin=0)
            if save_to_file:
                pyplot.savefig(title + 'dis.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
            pyplot.plot(range(1, 1 + len(self.total_omega)),
                        remove_infinities(self.total_omega))
            pyplot.xlabel('Iteration Step')
            pyplot.ylabel(r'$\Omega / \mathrm{\AA}^2$')
            pyplot.suptitle(title)
            pyplot.ylim(ymin=0)
            if save_to_file:
                pyplot.savefig(title + 'minim.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
            n = len(self.delta_omega_i)
            pdelta_omega_i = [d if d > 0 else 0 for d in
                              remove_infinities(self.delta_omega_i)]
            ndelta_omega_i = [-d if d < 0 else 0 for d in
                              remove_infinities(self.delta_omega_i)]
            pyplot.plot(range(1, 1 + n), pdelta_omega_i, 'b')
            pyplot.plot(range(1, 1 + n), ndelta_omega_i, 'r')
            pyplot.xlabel('Iteration Step')
            pyplot.ylabel(r'$\Delta{\Omega}_I / \mathrm{\AA}^2$')
            pyplot.suptitle(title)
            pyplot.semilogy()
            if save_to_file:
                pyplot.savefig(title + 'ddis.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
            n = len(self.delta_total_omega)
            pdelta_omega = [d if d > 0 else 0 for d in
                            remove_infinities(self.delta_total_omega)]
            ndelta_omega = [-d if d < 0 else 0 for d in
                            remove_infinities(self.delta_total_omega)]
            pyplot.plot(range(1, 1 + n), pdelta_omega, 'b')
            pyplot.plot(range(1, 1 + n), ndelta_omega, 'r')
            pyplot.xlabel('Iteration Step')
            pyplot.ylabel(r'$\Delta\Omega / \mathrm{\AA}^2$')
            pyplot.suptitle(title)
            pyplot.semilogy()
            if save_to_file:
                pyplot.savefig(title + 'dmin.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()
            n = len(self.spreads)
            spreads = array(self.spreads).T
            n_wannier = len(spreads)
            for w in range(n_wannier):
                pyplot.plot(range(n), remove_infinities(spreads[w]),
                            label=str(w))
            pyplot.xlabel('Iteration Step')
            pyplot.ylabel(r'$\Delta{\Omega}_w / \mathrm{\AA}^2$')
            pyplot.suptitle(title)
            pyplot.ylim(ymin=0)
            pyplot.legend()
            if save_to_file:
                pyplot.savefig(title + 'spreads.pdf')
            else:
                pyplot.show()
            pyplot.delaxes()

    def plot_from_file(self, filename, title='', save_to_file=False):
        self.read_from_file(filename)
        self.plot(title, save_to_file)
        self.plot_md(title, save_to_file)
