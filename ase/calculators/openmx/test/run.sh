#!/bin/bash
#SBATCH -J dos    # jobname
#SBATCH -p normal      # fast, normal
#SBATCH -N 1         # number of node
#SBATCH -o test.o%j  # stdout
#SBATCH -e test.e%j  # stderr
#SBATCH -c 16         # cpu per task. 
#SBATCH -x i[01-09,14]   # name of node to be ignored

#openmx=/group1/hermite/program/openmx3.8/work/openmx
#input=FeCS.dat#
dos=/group1/hermite/program/dos_openmx/work/projdos
dosin=$PWD/projinput
srun $dos $PWD/FeCS-gga.scfout $dosin
