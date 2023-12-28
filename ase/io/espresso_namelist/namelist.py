import re
import warnings
from pathlib import Path

from .keys import ALL_KEYS


class Namelist(dict):

    KEYS = ALL_KEYS["pw"]

    def __init__(self, __input_data=None, **kwargs):
        super().__init__()
        if __input_data is None:
            __input_data = {}
        self.update(self._lower_keys(dict(__input_data, **kwargs)))

    def _lower_keys(self, input_dict):
        return {
            k.lower(): self._lower_keys(v) if isinstance(v, dict) else v
            for k, v in input_dict.items()
        }

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Namelist(value)
        if not isinstance(key, str):
            raise TypeError(f"Key must be str, not {type(key).__name__}")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        super().__delitem__(key.lower())

    def __repr__(self):
        return "".join(self.to_string())

    @staticmethod
    def search_key(to_find, keys):
        """Search for a key in the namelist, case-insensitive.
        Returns the section and key if found, None otherwise.
        """
        for section in keys:
            for key in keys[section]:
                if re.match(rf"({key})\b(\(+.*\)+)?$", to_find):
                    return section, key

    def to_string(self):
        """Format a Namelist object as a string for writing to a file.
        Assume sections are ordered (taken care of in namelist construction)
        and that repr converts to a QE readable representation (except bools)

        Parameters
        ----------
        parameters : Namelist | dict
            Expecting a nested or flat dictionary of sections and key-value data.

        Returns
        -------
        pwi : List[str]
            Input line for the namelist
        """
        pwi = []
        for key, value in self.items():
            if isinstance(value, dict):
                pwi.append(f"&{key.upper()}\n")
                pwi.extend(Namelist.to_string(value))
                pwi.append("/\n")
            else:
                if value is True:
                    pwi.append(f"   {key:16} = .true.\n")
                elif value is False:
                    pwi.append(f"   {key:16} = .false.\n")
                elif isinstance(value, Path):
                    pwi.append(f'   {key:16} = "{value}"\n')
                else:
                    pwi.append(f"   {key:16} = {value!r}\n")
        return pwi

    def construct_namelist(self, binary='pw', warn=False, **kwargs):
        """
        Construct an ordered Namelist containing all the parameters given (as
        a dictionary or kwargs). Keys will be inserted into their appropriate
        section in the namelist and the dictionary may contain flat and nested
        structures. Any kwargs that match input keys will be incorporated into
        their correct section. All matches are case-insensitive, and returned
        Namelist object is a case-insensitive dict.

        If a key is not known to ase, but in a section within `parameters`,
        it will be assumed that it was put there on purpose and included
        in the output namelist. Anything not in a section will be ignored (set
        `warn` to True to see ignored keys).

        Keys with a dimension (e.g. Hubbard_U(1)) will be incorporated as-is
        so the `i` should be made to match the output.

        The priority of the keys is:
            kwargs[key] > parameters[key] > parameters[section][key]
        Only the highest priority item will be included.

        Parameters
        ----------
        parameters: dict
            Flat or nested set of input parameters.
        keys: Namelist | dict
            Namelist to use as a template for the output.
        warn: bool
            Enable warnings for unused keys.

        Returns
        -------
        input_namelist: Namelist
            espresso compatible namelist of input parameters.

        """

        keys = ALL_KEYS[binary]
        keys = Namelist(keys)

        constructed_namelist = {
            section: self.pop(
                section,
                {}) for section in keys}

        unused_keys = []
        for arg_key in list(self):
            search = Namelist.search_key(arg_key, keys)
            value = self.pop(arg_key)
            if search:
                section, key = search
                constructed_namelist[section][arg_key] = value
            else:
                unused_keys.append(arg_key)

        for arg_key in list(kwargs):
            search = Namelist.search_key(arg_key, keys)
            value = kwargs.pop(arg_key)
            if search:
                section, key = search
                constructed_namelist[section][arg_key] = value
            else:
                unused_keys.append(arg_key)

        if warn and unused_keys:
            warnings.warn("Unused keys: {}".format(", ".join(unused_keys)))

        self.update(constructed_namelist)
