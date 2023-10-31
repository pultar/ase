import os
import configparser
from collections.abc import Mapping
from pathlib import Path

from ase.utils import lazymethod
import shlex

ASE_CONFIG_FILE = Path.home() / ".config/ase/ase.conf"


class Config(Mapping):

    def __init__(self):
        self._dct = os.environ

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, item):
        if item in self.parser:
            return self.parser[item]
        elif item in self._dct:
            return self._dct[item]
        else:
            raise KeyError

    def __len__(self):
        return len(self._dct)

    def __contains__(self, item):
        return item in self._dct or item in self.parser

    @lazymethod
    def _paths_and_parser(self):
        def argv_converter(argv):
            return shlex.split(argv)

        parser = configparser.ConfigParser(converters={"argv": argv_converter})
        envpath = os.environ.get("ASE_CONFIG_PATH")
        if envpath is not None:
            paths = [Path(p) for p in envpath.split(":")]
        else:
            paths = [ASE_CONFIG_FILE, ]
        loaded_paths = parser.read(paths)
        return loaded_paths, parser

    @property
    def paths(self):
        return self._paths_and_parser()[0]

    @property
    def parser(self):
        return self._paths_and_parser()[1]

    def check_calculators(self):
        from ase.calculators.names import names, templates

        print("Calculators")
        print("===========")
        print()
        print("Configured in ASE")
        print("   |  Installed on machine")
        print("   |   |  Name & version")
        print("   |   |  |")
        for name in names:
            # configured = False
            # installed = False
            template = templates.get(name)
            # if template is None:
            # XXX no template for this calculator.
            # We need templates for all calculators somehow,
            # but we can probably generate those for old FileIOCalculators
            # automatically.
            #    continue

            fullname = name
            try:
                codeconfig = self.parser[name]
            except KeyError:
                codeconfig = None
                version = None
            else:
                if template is None:
                    codeconfig = None  # XXX we should not be executing this
                    version = None
                else:
                    profile = template.load_profile(codeconfig)
                    # XXX should be made robust to failure here:
                    version = profile.version()
                    fullname = f"{name}-{version}"

            def tickmark(thing):
                return "[ ]" if thing is None else "[x]"

            msg = "  {configured} {installed} {fullname}".format(
                configured=tickmark(codeconfig),
                installed=tickmark(version),
                fullname=fullname,
            )
            print(msg)

    def print_everything(self):
        print("Configuration")
        print("-------------")
        print()
        if not self.paths:
            print("No configuration loaded.")

        for path in self.paths:
            print(f"Loaded: {path}")

        print()
        for name, section in self.parser.items():
            print(name)
            if not section:
                print("  (Nothing configured)")
            for key, val in section.items():
                print(f"  {key}: {val}")
            print()

    def as_dict(self):
        return {key: dict(val) for key, val in self.parser.items()}


cfg = Config()
