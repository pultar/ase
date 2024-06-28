import re
import warnings
from pathlib import Path
from ase.io.espresso_namelist.keys import pwmat_keys
from ase.io.espresso import Namelist

class Namelist_pwmat(Namelist):

    def to_string(self, list_form=False):
        etot_input = []
        for key, value in self.items():
            if value is True or value == 'T':
                etot_input.append(f'{key} = T\n')
            elif value is False or value == 'F':
                etot_input.append(f'{key} = F\n')
            elif key.upper() == 'PARALLEL':
                etot_input.append('{}   {}\n'.format(*value))
            else:
                etot_input.append(f'{key} = {value}\n')
        if list_form:
            return etot_input
        else:
            return "".join(etot_input)

    def to_nested(self, warn=False, sorted_keys=False, **kwargs):
        keys = pwmat_keys
        unused_keys = []
        constructed_namelist = {}
        for arg_key in list(self):
            if arg_key.upper() in keys:
                value = self.pop(arg_key)
                constructed_namelist[arg_key.upper()] = value
            else:
                unused_keys.append(arg_key)
        for arg_key in list(kwargs):
            if arg_key.upper() in keys:
                value = kwargs.pop(arg_key)
                constructed_namelist[arg_key.upper()] = value
            else:
                unused_keys.append(arg_key)
        if unused_keys and warn:
            warnings.warn(f'Unused keys: {', '.join(unused_keys)}')
            
        if sorted_keys:
            constructed_namelist = dict(sorted(constructed_namelist.items()
                                               , key=lambda item:keys.index(item[0])))
        super().update(Namelist_pwmat(constructed_namelist))
