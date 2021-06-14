"""Input file creator for ph.x that works as closely as possible to Atomic Simulation Environment conventions.
The following arguments are not implemented:
    "dvscf_star",
    "drho_star",
"""
import numpy as np
import os


str_keys = {
    "title_line",
    "outdir",
    "prefix",
    "verbosity",
    "fildyn",
    "fildrho",
    "fildvscf",
    "electron_phonon",
    "ahc_dir",
    "diagonalization",
    "wpot_dir",
}

int_keys = {
    "niter_ph",
    "nmix_ph",
    "el_ph_nsigma",
    "ahc_nbnd",
    "ahc_nbndskip",
    "nq1",
    "nq2",
    "nq3",
    "nk1",
    "nk2",
    "nk3",
    "k1",
    "k2",
    "k3",
    "start_irr",
    "last_irr",
    "nat_todo",
    "modenum",
    "start_q",
    "last_q",
}

float_keys = {
    "tr2_ph",
    "eth_rps",
    "eth_ns",
    "dek",
    "el_ph_sigma",
}

list_float_keys = {
    "amass",
    "alpha_mix",
}

bool_keys = {
    "reduce_io",
    "epsil",
    "lrpa",
    "lnoloc",
    "trans",
    "lraman",
    "recover",
    "low_directory_check",
    "only_init",
    "qplot",
    "q2d",
    "q_in_band_form",
    "skip_upperfan",
    "shift_q",
    "zeu",
    "zue",
    "elop",
    "fpol",
    "ldisp",
    "nogg",
    "asr",
    "ldiag",
    "lqdir",
    "search_sym",
    "read_dns_bare",
    "ldvscf_interpolate",
    "do_long_range",
    "do_charge_neutral",
}


def bool_to_fortbool(x):
    if x:
        return ".true."
    elif not x:
        return ".false."
    else:
        raise Exception


def float_to_fortstring(x):
    return f"{x:0.07e}".replace("e", "d")


def write(directory, infilename: str = 'phonon.in', require_valid_calculation: bool = True, **kwargs):
    """Writes a ph.x input file when using the .write() method.
    All input arguments except STRUCTURE types are supported,
    but the input sanitation/validation is currently weak.

    Parameters
    ----------
    directory : str
        The directory in which the pw.x calculation has been done.
    infilename : str, optional
        The ph.x input file, by default "phonon.in"
    require_valid_calculation: bool, optional
        If True, throws an error if a valid SCF calculation directory does not exist, by default True
    """
    inputfile_name: str = directory + "/" + infilename
    if not os.path.exists(directory) and require_valid_calculation:
        raise FileNotFoundError(
            r"The calculation directory does not exist! \
            Make sure you have carried out a pw.x calculation before this, \
            and that the directory names are exactly equal."
        )

    with open(inputfile_name, "w") as fd:
        fd.write("&inputph\n")

        for key, value in kwargs.items():
            if key in bool_keys:
                assert type(value) == bool
                fd.write("=".join([key, bool_to_fortbool(value)]) + ",\n")

            if key in float_keys:
                assert type(value) == float
                fd.write("=".join([key, float_to_fortstring(value)]) + ",\n")

            if key in int_keys:
                assert type(value) == int
                fd.write(key + "=" + f"{value}" + ",\n")

            if key in str_keys:
                assert type(value) == str
                fd.write(key + "=" + f"'{value}'" + ",\n")

            if key in list_float_keys:
                assert type(value) == list
                for i, value in enumerate(value):
                    fd.write(
                        "=".join([key + f"({i+1})", float_to_fortstring(value)])
                        + ",\n"
                    )

        fd.write("/\n")