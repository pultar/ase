import time


def get_file_header(filename="geometry.in", parameters=None):
    """return the aims control/geometry.in header"""
    header = """
#=======================================================
# FHI-aims file: {}
# Created using the Atomic Simulation Environment (ASE)
# {}
#=======================================================
""".format(
        filename, time.asctime()
    )

    if parameters is not None:
        header += "# \n# List of parameters used to initialize the calculator:"
        for p, v in parameters.items():
            s = "#     {} : {}\n".format(p, v)
            header += s

    return header
