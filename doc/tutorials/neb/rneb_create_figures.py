# creates: reflective_path.png, rmineb.png
# creates: rneb_I.png, rneb_F.png
from ase.io import write, read
import runpy

runpy.run_path('rneb_reflective.py')
runpy.run_path('rneb_rmineb.py')

images = read('neb.traj@-5:')
for name, a in zip('IF', images[::len(images)-1]):
    cell = a.get_cell()
    del a.constraints
    a = a * (2, 2, 1)
    a.set_cell(cell)
    write('rneb-%s.pov' % name, a,
          povray_settings={'transparent':False, 'display':False})
