from __future__ import print_function
from ase.utils import basestring

# Copyright (C) 2010 by CAMd, DTU
# Please see the accompanying LICENSE file for further information.

# This file is taken (almost) verbatim from CMR with D. Landis agreement

FIELD_SEPARATOR = "\\"
PARA_START = "\n\n"
PARA_END = "\\\\@"

names = ['', '', 'Computer_system', 'Type_of_run', 'Method', 'Basis_set',
         'Chemical_formula', 'Person', 'Date', '', '', '', '', 'Title', '']
names_compact = ['', '', 'Computer_system', 'Type_of_run', 'Method',
                 'Basis_set', 'Chemical_formula', 'Person', 'Date', '', '', '',
                 '', 'Title', '']

charge_multiplicity = 15


class GaussianReader:

    def auto_type(self, data):
        """ tries to determine type"""
        try:
            return float(data)
        except ValueError:
            pass

        try:
            ds = data.split(",")
            array = []

            for d in ds:
                array.append(float(d))

            return array
        except ValueError:
            pass

        return data

    def __init__(self, filename, read_images=False):
        """filename is NOT optional"""
        if isinstance(filename, basestring):
            fileobj = open(filename, 'r')
        elif hasattr(filename,'seek'):
            fileobj = filename
            fileobj.seek(0)  # Re-wind fileobj
        else:
            msg = 'Cannot use given filename, make sure it is a string or a fileobject'
            raise RuntimeError(msg)

        content = fileobj.read()

# handles the case that users used windows after the calculation:
        content = content.replace("\r\n", "\n")

        self.parse(content)

        #read images from file
        if read_images:
            self.images = self.get_images(content)


    def get_images(self, content=None):
        """Read Images and return them or return them if already read"""
        if hasattr(self,'images'):
            return self.images
        elif content is None:
            raise RuntimeError('Images not available and no content parsed!')

        from ase.data import atomic_numbers
        from ase.atoms import Atoms
        from ase.atom import Atom
        images = []
        temp_items = content.split('Standard orientation')[1:]
        for item_i in temp_items:
            lines = [ line for line in item_i.split('\n') if len(line) > 0 ]
            #first 5 lines are headers
            del lines[:5]
            images.append(Atoms())
            for line in lines:
                #if only - in line it is the end
                if set(line).issubset(set('- ')):
                    break
                tmp_line = line.strip().split()
                if not len(tmp_line) == 6:
                    raise RuntimeError('Length of line does not match structure!')

                #read atom
                try:
                    atN = int(tmp_line[1])
                    pos = tuple(float(x) for x in tmp_line[3:])
                except ValueError:
                    raise ValueError('Expected a line with three integers and three floats.')
                images[-1].append(Atom(atN,pos))
        return images



    def parse(self, content):
        from ase.data import atomic_numbers
        self.data = []
        temp_items = content.split(PARA_START)
        seq_count = 0
        for i in temp_items:
            i = i.replace("\n ", "")
            if i.endswith(PARA_END):
                i = i.replace(PARA_END, "")
                i = i.split(FIELD_SEPARATOR)

                new_dict = {}
                self.data.append(new_dict)

                new_dict['Sequence number'] = seq_count
                seq_count += 1
                for pos in range(len(names)):
                    if names[pos] != "":
                        new_dict[names[pos]] = self.auto_type(i[pos])

                chm = i[charge_multiplicity].split(",")
                new_dict["Charge"] = int(chm[0])
                new_dict["Multiplicity"] = int(chm[1])

# Read atoms
                atoms = []
                positions = []
                position = charge_multiplicity + 1
                while position < len(i) and i[position] != "":
                    s = i[position].split(",")
                    atoms.append(atomic_numbers[s[0].capitalize()])
                    positions.append([float(s[1]), float(s[2]), float(s[3])])
                    position = position + 1

                new_dict["Atomic_numbers"] = atoms
                new_dict["Positions"] = positions
# Read more variables
                position += 1
                while position < len(i) and i[position] != "":
                    s = i[position].split('=')
                    if len(s) == 2:
                        new_dict[s[0]] = self.auto_type(s[1])
                    else:
                        print("Warning: unexpected input ", s)
                    position = position + 1

    def __iter__(self):
        """returns an iterator that iterates over all keywords"""
        return self.data.__iter__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, pos):
        return self.data[pos]
