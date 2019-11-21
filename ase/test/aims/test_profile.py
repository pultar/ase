from warnings import warn
from ase.calculators.aims import AimsProfile


def test_instantiation():
    AimsProfile()


def test_available():
    assert AimsProfile().available()


def test_version():
    version = AimsProfile().get_version()
    if version is None:
        warn("Version could not be found.")
    else:
        assert isinstance(version, str)


if __name__ == "__main__":
    test_instantiation()
    test_available()
    test_version()
