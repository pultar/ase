from collections.abc import Mapping
from ase.utils import lazyproperty
from typing import Dict


class Listing(Mapping):
    """ Class, that lists something, e.g. Plugins or Plugables
    (of calculators or formats etc...).
    The added items are required to have a name attribute.
    """

    def __init__(self):
        self.items = {}

    def add(self, item):
        """ Add an item """
        self.items[item.name] = item

    def info(self, prefix: str = '', opts: Dict = {}) -> str:
        """
        Parameters
        ----------
        prefix
            Prefix, which should be prepended before each line.
            E.g. indentation.
        opts
            Dictionary, that can holds options,
            what info to print and which not.

        Returns
        -------
        info
          Information about the object and (if applicable) contained items.
        """

        out = [i.info(prefix) for i in self.sorted]
        return '  \n'.join(out)

    @staticmethod
    def _sorting_key(i):
        return i.name.lower()

    @lazyproperty
    def sorted(self):
        """ Return items in the listing, sorted by a predefined criteria """
        ins = list(self.items.values())
        ins.sort(key=self._sorting_key)
        return ins

    def __len__(self):
        return len(self.items)

    def __getitem__(self, name):
        out = self.find_by_name(name)
        if not out:
            raise KeyError(f"There is no {name} in {self}")
        return out

    def __iter__(self):
        return iter(self.items.values())

    def find_by(self, attribute, value):
        """ Find plugin according the given attribute.
        The attribute can be given by list of alternative values,
        or not at all - in this case, the default value for the attribute
        will be used """
        for i in self:
            if Listing._item_has_attribute(i, attribute, value):
                return i

    def find_all_by(self, attribute, value):
        """ Find plugin according the given attribute.
        The attribute can be given by list of alternative values,
        or not at all - in this case, the default value for the attribute
        will be used """
        return (i for i in self if
                Listing._item_has_attribute(i, attribute, value))

    @staticmethod
    def _item_has_attribute(obj, attribute, value):
        v = getattr(obj, attribute, None)
        if value == v:
            return True
        if isinstance(v, (list, set)):
            for i in v:
                if i == value:
                    return True
        return False

    def find_by_name(self, name):
        return self.items.get(name, None)
