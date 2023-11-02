from collections.abc import Iterable


def write_incar(directory, parameters):
    incar_string = generate_incar_lines(parameters)
    with open(f"{directory}/INCAR", "w") as incar:
        incar.write(incar_string)


def generate_incar_lines(parameters):
    if isinstance(parameters, str):
        return parameters
    elif parameters is None:
        return ""
    # check for empty dict
    elif not parameters:
        return ""
    else:
        incar_lines = []
        for item in parameters.items():
            incar_lines += list(generate_line(*item))
        return "\n".join(incar_lines) + "\n"


def generate_line(key, value, num_spaces=1):
    indent = " " * num_spaces
    if isinstance(value, str):
        if value.find("\n") != -1:
            value = '"' + value + '"'
        yield indent + f"{key.upper()} = {value}"
    elif isinstance(value, dict):
        yield indent + f"{key.upper()} {{"
        for item in value.items():
            yield from generate_line(*item, num_spaces + 4)
        yield indent + "}"
    elif isinstance(value, Iterable):
        yield indent + f"{key.upper()} = {' '.join(str(x) for x in value)}"
    else:
        yield indent + f"{key.upper()} = {value}"
