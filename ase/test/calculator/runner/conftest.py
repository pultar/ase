import pytest

def pytest_addoption(parser):
    parser.addoption("--executable", action="store")


@pytest.fixture(scope='session')
def executable(request):
    executable_value = request.config.option.executable
    if executable_value is None:
        pytest.skip()
    return executable_value