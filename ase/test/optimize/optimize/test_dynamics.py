from itertools import product
import math
from typing import Any, Callable, Dict, List, Tuple
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

    @staticmethod
    def test_should_attach_callback_function():
        ...

    @staticmethod
    def test_should_raise_not_implemented_error_when_calling_dynamics_todict(
        dynamics: Dynamics, observer: Dict[str, Any]
    ) -> None:
        with pytest.raises(NotImplementedError):
            _ = dynamics.todict()


class TestCallObservers:
    @staticmethod
    @pytest.fixture(name="insert_observers")
    def fixture_insert_observers(
        dynamics: Dynamics,
    ) -> Callable[[List[int]], List[Tuple[Callable, int, int, str]]]:
        inserted_observers: List[Tuple[Callable, int, int, str]] = []

        def _insert_observer(*, intervals) -> List[Tuple[Callable, int, int, str]]:
            for i, interval in enumerate(intervals):
                observer = (print, i, interval, f"Observer {i}")
                inserted_observers.append(observer)
                dynamics.insert_observer(*observer)

            return inserted_observers

        return _insert_observer

    @staticmethod
    def test_should_call_inserted_observers(
        dynamics: Dynamics,
        capsys: pytest.CaptureFixture,
        insert_observers: Callable[[List[int]], List[Tuple[Callable, int, int, str]]],
    ) -> None:
        _ = insert_observers(intervals=[1])
        dynamics.call_observers()
        output: str = capsys.readouterr().out
        lines = output.splitlines()
        observer_outputs_present = []
        for i, _ in enumerate(dynamics.observers):
            observer_outputs_present.append(f"Observer {i}" in lines[i])
        assert all(observer_outputs_present)

    @staticmethod
    @pytest.mark.parametrize(
        "interval,step", [(i, i * j) for i, j in product([1, 2], repeat=2)]
    )
    def test_should_call_observer_every_positive_interval(
        dynamics: Dynamics,
        capsys: pytest.CaptureFixture,
        insert_observers: Callable[[List[int]], List[Tuple[Callable, int, int, str]]],
        interval: int,
        step: int,
    ) -> None:
        _ = insert_observers(intervals=[interval])
        dynamics.nsteps = step
        dynamics.call_observers()
        output: str = capsys.readouterr().out
        lines = output.splitlines()
        observer_outputs_present = []
        for i, _ in enumerate(dynamics.observers):
            observer_outputs_present.append(f"Observer {i}" in lines[i])
        assert all(observer_outputs_present)

    @staticmethod
    @pytest.mark.parametrize(
        "interval,step", [(i, i * j + 1) for i, j in product([-1, 2, 3], [1, 2])]
    )
    def test_should_not_call_observer_if_not_specified_interval(
        dynamics: Dynamics,
        capsys: pytest.CaptureFixture,
        insert_observers: Callable[[List[int]], List[Tuple[Callable, int, int, str]]],
        interval: int,
        step: int,
    ) -> None:
        _ = insert_observers(intervals=[interval])
        dynamics.nsteps = step
        dynamics.call_observers()
        output: str = capsys.readouterr().out
        lines = output.splitlines()
        observer_outputs_present = []
        for i, _ in enumerate(dynamics.observers):
            observer_outputs_present.append(
                all(f"Observer {i}" not in line for line in lines)
            )
        assert all(observer_outputs_present)

    @staticmethod
    @pytest.mark.parametrize("interval", (0, -1, -2, -3))
    def test_should_call_observer_on_specified_step_if_interval_negative(
        dynamics: Dynamics,
        capsys: pytest.CaptureFixture,
        insert_observers: Callable[[List[int]], List[Tuple[Callable, int, int, str]]],
        interval: int,
    ) -> None:
        _ = insert_observers(intervals=[interval])
        dynamics.nsteps = abs(interval)
        dynamics.call_observers()
        output: str = capsys.readouterr().out
        lines = output.splitlines()
        observer_outputs_present = []
        for i, _ in enumerate(dynamics.observers):
            observer_outputs_present.append(
                all(f"Observer {i}" not in line for line in lines)
            )
        assert all(observer_outputs_present)

    @staticmethod
    def test_should_call_observers_in_order_of_position_in_list(
        dynamics: Dynamics,
        capsys: pytest.CaptureFixture,
        insert_observers: List[Tuple[Callable, int, int, str]],
        interval: int,
    ) -> None:
        # should have multiple observers with interval == 1
        # nsteps should be 0
        dynamics.nsteps = math.abs(interval)
        dynamics.call_observers()
        captured: str = capsys.readouterr().out
        messages_in_order = []
        for i, line in enumerate(captured.splitlines()):
            message = dynamics.observers[i][2][0]
            messages_in_order.append(line == message)

        assert all(messages_in_order)

    @staticmethod
    def test_should_call_attached_callback_function():
        ...
