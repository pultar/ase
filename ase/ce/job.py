import os
from textwrap import dedent
from ase.db import connect
from ase.io import write
from subprocess import check_output
from ase.ce import jobscripts

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


class Submit(object):
    """
    Class that handles submission of DFT computations to the batch system.
    """
    def __init__(self, db_name, compute_param, num_cores):
        self.db_name = db_name
        self.compute_param = compute_param
        self.num_cores = num_cores
        self.db = connect(self.db_name)

    @property
    def submit_new(self):
        """
        Find a row in the database that has the condition 'queued=False' and
        'started=False' and submit it to a SLURM batch system.
        """
        param = self.compute_param
        db_name = self.db_name
        condition = 'queued=False, started=False'
        name = [row.name for row in self.db.select(condition)][0]
        atoms = self.db.get_atoms(name=name)
        id = self.db.get(name=name).id
        suffix_length = len(name.split('_')[-1]) + 1
        prefix = name[:(-1*suffix_length)]
        self.remove_X(atoms)
        param['system'] = name
        param['istart'] = 0

        if not os.path.exists('./%s/%s' %(prefix, name)):
            os.makedirs('./%s/%s' %(prefix, name))
            os.system('chmod -R 777 ./%s/%s' %(prefix, name))

        with cd('./%s/%s' %(prefix, name)):
            # write traj file
            write('input.traj', atoms)
            vasp_file = open('vasp.py', 'w')
            print >> vasp_file, dedent(jobscripts.vasp_new(param, name, db_name))
            vasp_file.close()

            run_file = open('run.sh', 'w')
            if self.num_cores == 8:
                print >> run_file, dedent(jobscripts.slurm_script_8(name))
            elif self.num_cores == 16:
                print >> run_file, dedent(jobscripts.slurm_script_16(name))
            else:
                print >> run_file, dedent(jobscripts.slurm_script_24(name))
            run_file.close()

            os.system('chmod u+xwr vasp.py run.sh')
            output_string = check_output(['sbatch run.sh'], shell=True)


        # wait and get the response to confirm
        if "Submitted" not in output_string:
            raise ValueError("Job name %s may not be submitted" %name)

        print('submitted %s' %name)
        self.db.update(id, queued=True)
        return True

    @property
    def submit_restart(self):
        param = self.compute_param
        db_name = self.db_name
        condition = 'converged=False, started=True'
        names = [row.name for row in self.db.select(condition)]
        queue_list = self.jobs_in_queue
        name = [i for i in names if i not in queue_list][0]
        id = self.db.get(name=name).id
        param['system'] = name
        print(name)
        print([i for i in names if i not in queue_list])
        print(len([i for i in names if i not in queue_list]))
        suffix_length = len(name.split('_')[-1]) + 1
        prefix = name[:(-1*suffix_length)]
        with cd('./%s/%s' %(prefix, name)):
            # Check the reason for incompletion
            logfile = open('%s.log' %name, 'r')
            log_msg = logfile.read()
            logfile.close()
            # Case where the last job halted due to the time limit
            if "TIME LIMIT" in log_msg:
                print('Job previously halted due to the walltime limit.')
                print('Resuming with regular conditions.')
            # Case where there was some error in the previous run
            else:
                rhosyg_error = "RHOSYG: internal error: stars are not distinct"
                zbrent_error = "ZBRENT: fatal error in bracketing"
                pricel_error = "VERY BAD NEWS! internal error in subroutine PRICEL"
                vaspout = open('vasp.out', 'r')
                vaspout_msg = vaspout.read()
                vaspout.close()
                print('Job previously halted due to an error:')
                if rhosyg_error in vaspout_msg:
                    print('----------------------------------------------')
                    print(rhosyg_error)
                    print('----------------------------------------------')
                    print('Setting ISYM = 0 to circumvent the issue.')
                    param['isym'] = 0
                elif zbrent_error in vaspout_msg:
                    print('--------------------------------------')
                    print(zbrent_error)
                    print('--------------------------------------')
                    print('Setting IBRION = 1 to circumvent the issue')
                    param['ibrion'] = 1
                elif pricel_error in vaspout_msg:
                    print('--------------------------------------')
                    print(pricel_error)
                    print('--------------------------------------')
                    print('Rattle the structure to break symmetry')
                    raise ValueError('Need to rattle separately')
                else:
                #    param['isym'] = 0
                #    param['ibrion'] = 1
                    raise ValueError('Halted due to an unknown error')

            vasp_file = open('vasp.py', 'w')
            print >> vasp_file, dedent(jobscripts.vasp_restart(param, name, db_name))
            vasp_file.close()

            os.system('chmod u+xwr vasp.py run.sh')
            output_string = check_output(['sbatch run.sh'], shell=True)

        # wait and get the response to confirm
        if "Submitted" not in output_string:
            raise ValueError("Job name %s may not be submitted" %name)

        print('submitted %s' %name)
        self.db.update(id, queued=True, converged=False, started=False)
        return True


    def remove_X(self, atoms):
        """
        Vacancies are specified with the ghost atom 'X', which must be removed
        before passing the atoms object to a calculator.
        """
        del atoms[[atom.index for atom in atoms if atom.symbol=='X']]
        return True


    @property
    def jobs_in_queue(self):
        """
        Returns a list of job names that are in the SLURM batch system.
        """
        job_names = []
        jobs_string = check_output(['qstat -f | grep -C 1 $USER' ], shell=True).splitlines()
        for line in jobs_string:
            if 'Job_Name' not in line:
                continue
            line = line.split()
            job_names.append(line[-1])
        return job_names

