# creates: fecn6.png
atoms = ''  # flake urself
write = ''  # will be overwritten
exec(compile(open('build_fecn6.py').read(), 'build_fecn6.py', 'exec'))
atoms.rotate(45, (0, 1, 1))
write('fecn6.png', atoms)
