from typing import Any, Dict
import pytest

from ase import Atoms
from ase.optimize.optimize import Dynamics


class DummyDynamics(Dynamics):
    def step(self):
        ...

    def log(self):
        ...


@pytest.fixture(name="dynamics")
def fixture_dynamics(atoms: Atoms) -> Dynamics:
    return DummyDynamics(atoms=atoms)


class TestDynamics:
    @staticmethod
    @pytest.fixture(name="observer", scope="function")
    def fixture_observer() -> Dict[str, Any]:
        observer = {
            "function": print,
            "position": 0,
            "interval": 1,
        }
        return observer

    @staticmethod
    def test_should_return_zero_steps_after_instantiation(dynamics: Dynamics):
        assert dynamics.get_number_of_steps() == 0

    @staticmethod
    def test_should_insert_observer(
        observer: Dict[str, Any], dynamics: Dynamics
    ) -> None:
        dynamics.insert_observer(**observer)
        inserted_observer = dynamics.observers[observer["position"]]

        attributes_in_observer = []
        for k, v in observer.items():
            if k != "position":
                attributes_in_observer.append(v in inserted_observer)
        
        assert all(attributes_in_observer)

    @staticmethod
    def test_should_insert_observer_with_write_method_of_non_callable(
        dynamics: Dynamics, observer: Dict[str, Any]
    ) -> None:
        observer["function"] = dynamics.optimizable.atoms
        dynamics.insert_observer(**observer)
        assert dynamics.observers[0][0] == dynamics.optimizable.atoms.write


class TestCallObserver:
    @staticmethod
    def test_should_call_inserted_observer():
        ...

    @staticmethod
    def test_should_attach_callback_function():
        ...

    @staticmethod
    def test_should_call_attached_callback_function():
        ...

    @staticmethod
    def test_should_call_observer_every_positive_interval():
        ...

    @staticmethod
    def test_should_not_call_observer_if_not_specified_interval():
        ...

    @staticmethod
    def test_should_call_observer_only_once_if_interval_negative():
        ...

    @staticmethod
    def test_should_only_call_observer_on_negative_interval():
        ...
