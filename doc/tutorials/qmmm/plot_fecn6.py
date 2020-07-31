# creates: fecn6.png
import runpy
dct = runpy.run_path('build_fecn6.py')
atoms = dct['atoms']
write = dct['write']
atoms.rotate(45, (0, 1, 1))
write('fecn6.png', atoms)
