def vasp_restart(param, name, db_name):
    script_text="""\
    import os
    from ase.calculators.vasp import Vasp
    from ase.io import read, write
    from ase.db import connect
    from ase.io.trajectory import TrajectoryWriter
    from ase.calculators.singlepoint import SinglePointCalculator

    # update database
    name='"""+str(name)+"""'
    db_name='../../"""+str(db_name)+"""'
    con = connect(db_name)
    id = con.get(name=name).id
    con.update(id, started=True, queued=False)

    # save the OUTCAR from the previous run to traj
    outcar = read('OUTCAR', index=':')
    writer = TrajectoryWriter('output.traj', mode='a')
    for atoms in outcar:
        writer.write(atoms=atoms)

    # run calculation
    atoms=read('OUTCAR', -1)
    compute_param="""+str(param)+"""
    calc = Vasp(**compute_param)
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    # store in traj file
    outcar = read('OUTCAR', index=':')
    for atoms in outcar:
        writer.write(atoms=atoms)

    # run with the final structure to ensure convergence
    compute_param['ibrion'] = 1
    while len(outcar) > 1:
        atoms = read('OUTCAR', -1)
        calc = Vasp(**compute_param)
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        # append the image to traj file
        outcar = read('OUTCAR', index=':')
        for atoms in outcar:
            writer.write(atoms=atoms)

    # It is converged at this point -> update database
    # atoms object cannot be updated, copy and create new one with energy
    con.update(id, started='', queued='', converged=True)
    key_value_pairs = con.get(name=name).key_value_pairs
    atoms = con.get(name=name).toatoms()
    calc= SinglePointCalculator(atoms, energy=energy)
    atoms.set_calculator(calc)
    del con[id]
    con.write(atoms, key_value_pairs=key_value_pairs)
    os.system('rm WAVECAR')
    """
    return script_text

def vasp_new(param, name, db_name):
    script_text="""\
    import os
    from ase.calculators.vasp import Vasp
    from ase.io import read, write
    from ase.db import connect
    from ase.io.trajectory import TrajectoryWriter
    from ase.calculators.singlepoint import SinglePointCalculator

    def set_magmom(atoms):
        V_index = [a.index for a in atoms if a.symbol  == 'V']
        for V in V_index:
            atoms[V].magmom = 2.0
        Cr_index = [a.index for a in atoms if a.symbol  == 'Cr']
        for Cr in Cr_index:
            atoms[Cr].magmom = 5.0

    # update database
    name='"""+str(name)+"""'
    db_name='../../"""+str(db_name)+"""'
    con = connect(db_name)
    id = con.get(name=name).id
    con.update(id, started=True, queued=False)

    # run calculation
    atoms=read('input.traj')
    set_magmom(atoms)
    compute_param="""+str(param)+"""
    calc = Vasp(**compute_param)
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    # store in traj file
    outcar = read('OUTCAR', index=':')
    writer = TrajectoryWriter('output.traj', mode='w')
    for atoms in outcar:
        writer.write(atoms=atoms)


    # run with the final structure to ensure convergence
    compute_param['ibrion'] = 1
    atoms = read('OUTCAR', -1)
    calc = Vasp(**compute_param)
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    # append the image to traj file
    outcar = read('OUTCAR', index=':')
    writer = TrajectoryWriter('output.traj', mode='a')
    for atoms in outcar:
        writer.write(atoms=atoms)

    # It is converged at this point -> update database
    # atoms object cannot be updated, copy and create new one with energy
    con.update(id, started='', queued='', converged=True)
    key_value_pairs = con.get(name=name).key_value_pairs
    atoms = con.get(name=name).toatoms()
    calc= SinglePointCalculator(atoms, energy=energy)
    atoms.set_calculator(calc)
    del con[id]
    con.write(atoms, key_value_pairs=key_value_pairs)
    os.system('rm WAVECAR')
    """
    return script_text

def slurm_script_8(job_name):
    script_text="""\
    #!/bin/bash
    #SBATCH --mail-user=user@univ.edu
    #SBATCH --mail-type=ALL
    #SBATCH -N 2
    #SBATCH -n 16
    #SBATCH --time=150:00:00
    #SBATCH --output="""+str(job_name)+""".log
    #SBATCH --job-name="""+str(job_name)+"""
    #SBATCH --partition=xeon8

    python vasp.py"""
    return script_text

def slurm_script_16(job_name):
    script_text="""\
    #!/bin/bash
    #SBATCH --mail-user=user@univ.edu
    #SBATCH --mail-type=ALL
    #SBATCH -N 1
    #SBATCH -n 16
    #SBATCH --time=150:00:00
    #SBATCH --output="""+str(job_name)+""".log
    #SBATCH --job-name="""+str(job_name)+"""
    #SBATCH --partition=xeon16

    python vasp.py"""
    return script_text

def slurm_script_24(job_name):
    script_text="""\
    #!/bin/bash
    #SBATCH --mail-user=user@univ.edu
    #SBATCH --mail-type=ALL
    #SBATCH -N 1
    #SBATCH -n 24
    #SBATCH --time=50:00:00
    #SBATCH --output="""+str(job_name)+""".log
    #SBATCH --job-name="""+str(job_name)+"""
    #SBATCH --partition=xeon24

    python vasp.py"""
    return script_text
