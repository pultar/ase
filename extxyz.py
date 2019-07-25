from ase import Atoms
from ase.io import write
a = Atoms("He")
a.info['string_property'] = "abcdefgh"
a.info['scalar_property'] = 1234
a.info['list_property'] = [3.14, 2.71, 1.41, "ABC"]
from io import StringIO
s = StringIO()
write(s, [a], 'extxyz')

etalon="""1
Properties=species:S:1:pos:R:3 string_property=abcdefgh scalar_property=1234 list_property="3.14 2.71 1.41 ABC" pbc="F F F"
He       0.00000000       0.00000000       0.00000000"""
assert (s.getvalue() == etalon)

