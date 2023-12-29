import re
import warnings
from collections import UserDict
from collections.abc import MutableMapping
from pathlib import Path

from .keys import ALL_KEYS


class Namelist(UserDict):
    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __setitem__(self, key, value):
        super().__setitem__(
            key.lower(), Namelist(value) if isinstance(
                value, (dict, MutableMapping)) else value, )

    def __delitem__(self, key):
        super().__delitem__(key.lower())

    @staticmethod
    def search_key(to_find, keys):
        """Search for a key in the namelist, case-insensitive.
        Returns the section and key if found, None otherwise.
        """
        for section in keys:
            for key in keys[section]:
                if re.match(rf"({key})\b(\(+.*\)+)?$", to_find):
                    return section

    def to_string(self, indent=0, list_form=False):
        """Format a Namelist object as a string for writing to a file.
        Assume sections are ordered (taken care of in namelist construction)
        and that repr converts to a QE readable representation (except bools)

        Parameters
        ----------
        parameters : Namelist | dict
            Expecting a nested or flat dictionary of sections
            and key-value data.

        Returns
        -------
        pwi : List[str]
            Input line for the namelist
        """
        pwi = []
        for key, value in self.items():
            if isinstance(value, (Namelist, dict)):
                pwi.append(f"{' '* indent}&{key.upper()}\n")
                pwi.extend(Namelist.to_string(value, indent=indent + 3))
                pwi.append(f"{' '* indent}/\n")
            else:
                if value is True:
                    pwi.append(f"{' '* indent}{key:16} = .true.\n")
                elif value is False:
                    pwi.append(f"{' '* indent}{key:16} = .false.\n")
                elif isinstance(value, Path):
                    pwi.append(f"{' '* indent}{key:16} = '{value}'\n")
                else:
                    pwi.append(f"{' '* indent}{key:16} = {value!r}\n")
        if list_form:
            return pwi
        else:
            return "".join(pwi)

    def to_nested(self, binary='pw', warn=False, **kwargs):
        keys = ALL_KEYS[binary]

        constructed_namelist = Namelist({
            section: self.pop(section, {}) for section in keys
        })

        unused_keys = []
        for arg_key in list(self):
            section = Namelist.search_key(arg_key, keys)
            value = self.pop(arg_key)
            if section:
                constructed_namelist[section][arg_key] = value
            else:
                unused_keys.append(arg_key)

        for arg_key in list(kwargs):
            section = Namelist.search_key(arg_key, keys)
            value = kwargs.pop(arg_key)
            if section:
                constructed_namelist[section][arg_key] = value
            else:
                unused_keys.append(arg_key)

        if unused_keys and warn:
            warnings.warn(
                f"Unused keys: {', '.join(unused_keys)}",
                UserWarning,
            )

        super().update(constructed_namelist)
