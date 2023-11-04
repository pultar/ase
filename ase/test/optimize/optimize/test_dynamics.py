from typing import Callable
import pytest

from ase.optimize.optimize import Dynamics


class DummyDynamics:
    def step(self):
        ...

    def log(self):
        ...


@pytest.fixture(name="dynamics")
def fixture_dynamics() -> Dynamics:
    return DummyDynamics()


class TestDynamics:
    @staticmethod
    def fixture_observer() -> tuple[Callable, int, list, dict]:
        observer = (print, 1, None, None)
        return observer

    @staticmethod
    def test_should_return_zero_after_instantiation(dynamics: Dynamics):
        assert dynamics.get_number_of_steps() == 0

    # parametrize w.r.t. position
    @staticmethod
    def test_should_insert_observer(
        observer: tuple[Callable, int, list, dict], dynamics: Dynamics
    ) -> None:
        dynamics.insert_observer(*observer)
        assert dynamics.observers[0] == observer


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
