
class Namelist(OrderedDict):
    """Case insensitive dict that emulates Fortran Namelists."""

    def __contains__(self, key):
        return super().__contains__(key.lower())

    def __delitem__(self, key):
        return super().__delitem__(key.lower())

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)

    def get(self, key, default=None):
        return super().get(key.lower(), default)

PW_KEYS = {
    'CONTROL': [
        'calculation', 'title', 'verbosity', 'restart_mode', 'wf_collect',
        'nstep', 'iprint', 'tstress', 'tprnfor', 'dt', 'outdir', 'wfcdir',
        'prefix', 'lkpoint_dir', 'max_seconds', 'etot_conv_thr',
        'forc_conv_thr', 'disk_io', 'pseudo_dir', 'tefield', 'dipfield',
        'lelfield', 'nberrycyc', 'lorbm', 'lberry', 'gdir', 'nppstr',
        'lfcpopt', 'monopole'
    ],
    'SYSTEM': [
        'ibrav', 'nat', 'ntyp', 'nbnd', 'tot_charge', 'tot_magnetization',
        'starting_magnetization', 'ecutwfc', 'ecutrho', 'ecutfock', 'nr1',
        'nr2', 'nr3', 'nr1s', 'nr2s', 'nr3s', 'nosym', 'nosym_evc', 'noinv',
        'no_t_rev', 'force_symmorphic', 'use_all_frac', 'occupations',
        'one_atom_occupations', 'starting_spin_angle', 'degauss', 'smearing',
        'nspin', 'noncolin', 'ecfixed', 'qcutz', 'q2sigma', 'input_dft',
        'exx_fraction', 'screening_parameter', 'exxdiv_treatment',
        'x_gamma_extrapolation', 'ecutvcut', 'nqx1', 'nqx2', 'nqx3',
        'lda_plus_u', 'lda_plus_u_kind', 'Hubbard_U', 'Hubbard_J0',
        'Hubbard_alpha', 'Hubbard_beta', 'Hubbard_J',
        'starting_ns_eigenvalue', 'U_projection_type', 'edir',
        'emaxpos', 'eopreg', 'eamp', 'angle1', 'angle2',
        'constrained_magnetization', 'fixed_magnetization', 'lambda',
        'report', 'lspinorb', 'assume_isolated', 'esm_bc', 'esm_w',
        'esm_efield', 'esm_nfit', 'fcp_mu', 'vdw_corr', 'london',
        'london_s6', 'london_c6', 'london_rvdw', 'london_rcut',
        'ts_vdw_econv_thr', 'ts_vdw_isolated', 'xdm', 'xdm_a1', 'xdm_a2',
        'space_group', 'uniqueb', 'origin_choice', 'rhombohedral', 'zmon',
        'realxz', 'block', 'block_1', 'block_2', 'block_height'
    ],
    'ELECTRONS': [
        'electron_maxstep', 'scf_must_converge', 'conv_thr', 'adaptive_thr',
        'conv_thr_init', 'conv_thr_multi', 'mixing_mode', 'mixing_beta',
        'mixing_ndim', 'mixing_fixed_ns', 'diagonalization', 'ortho_para',
        'diago_thr_init', 'diago_cg_maxiter', 'diago_david_ndim',
        'diago_full_acc', 'efield', 'efield_cart', 'efield_phase',
        'startingpot', 'startingwfc', 'tqr'
    ],
    'IONS': [
        'ion_dynamics', 'ion_positions', 'pot_extrapolation',
        'wfc_extrapolation', 'remove_rigid_rot', 'ion_temperature', 'tempw',
        'tolp', 'delta_t', 'nraise', 'refold_pos', 'upscale', 'bfgs_ndim',
        'trust_radius_max', 'trust_radius_min', 'trust_radius_ini', 'w_1',
        'w_2'
    ],
    'CELL': [
        'cell_dynamics', 'press', 'wmass', 'cell_factor', 'press_conv_thr',
        'cell_dofree'
    ]
}

PH_KEYS = Namelist(
    {
        "INPUTPH": [
            "amass",
            "outdir",
            "prefix",
            "niter_ph",
            "tr2_ph",
            "alpha_mix",
            "nmix_ph",
            "verbosity",
            "reduce_io",
            "max_seconds",
            "dftd3_hess",
            "fildyn",
            "fildrho",
            "fildvscf",
            "epsil",
            "lrpa",
            "lnoloc",
            "trans",
            "lraman",
            "eth_rps",
            "eth_ns",
            "dek",
            "recover",
            "low_directory_check",
            "only_init",
            "qplot",
            "q2d",
            "q_in_band_form",
            "electron_phonon",
            "el_ph_nsigma",
            "el_ph_sigma",
            "ahc_dir",
            "ahc_nbnd",
            "ahc_nbndskip",
            "skip_upperfan",
            "lshift_q",
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
            "nq1",
            "nq2",
            "nq3",
            "nk1",
            "nk2",
            "nk3",
            "k1",
            "k2",
            "k3",
            "diagonalization",
            "read_dns_bare",
            "ldvscf_interpolate",
            "wpot_dir",
            "do_long_range",
            "do_charge_neutral",
            "start_irr",
            "last_irr",
            "nat_todo",
            "modenum",
            "start_q",
            "last_q",
            "dvscf_star",
            "drho_star",
        ]
    }
)

PP_KEYS = {
    'INPUTPP': [
        'prefix',
        'outdir',
        'filplot',
        'plot_num',
        'spin_component',
        'spin_component',
        'emin',
        'emax',
        'delta_e',
        'degauss_ldos',
        'use_gauss_ldos',
        'sample_bias',
        'kpoint',
        'kband',
        'lsign',
        'spin_component',
        'emin',
        'emax',
        'spin_component',
        'spin_component',
        'spin_component',
        'spin_component'
    ],
    'PLOT': [
        'nfile',
        'filepp',
        'weight',
        'iflag',
        'output_format',
        'fileout',
        'interpolation',
        'e1',
        'x0',
        'nx',
        'e1',
        'e2',
        'x0',
        'nx',
        'ny',
        'e1',
        'e2',
        'e3',
        'x0',
        'nx',
        'ny',
        'nz',
        'radius',
        'nx',
        'ny'
    ]
}

MATDYN_KEYS = {
    'INPUT': [
        'flfrc',
        'asr',
        'huang',
        'dos',
        'nk1',
        'nk2',
        'nk3',
        'deltaE',
        'ndos',
        'degauss',
        'fldos',
        'flfrq',
        'flvec',
        'fleig',
        'fldyn',
        'at',
        'l1',
        'l2',
        'l3',
        'ntyp',
        'amass',
        'readtau',
        'fltau',
        'la2F',
        'q_in_band_form',
        'q_in_cryst_coord',
        'eigen_similarity',
        'fd',
        'na_ifc',
        'nosym',
        'loto_2d',
        'loto_disable',
        'read_lr',
        'write_frc'
    ]
}

DYNMAT_KEYS = {
    'INPUT': [
        'fildyn',
        'q',
        'amass',
        'asr',
        'remove_interaction_blocks',
        'axis',
        'lperm',
        'lplasma',
        'filout',
        'fileig',
        'filmol',
        'filxsf',
        'loto_2d',
        'el_ph_nsig',
        'el_ph_sigma'
    ]
}

Q2R_KEYS = {
    'INPUT': [
        'fildyn',
        'flfrc',
        'zasr',
        'loto_2d',
        'write_lr'
    ]
}

DOS_KEYS = {
    "&DOS": [
        "prefix",
        "outdir",
        "bz_sum",
        "ngauss",
        "degauss",
        "emin",
        "emax",
        "deltae",
        "fildos"
    ]
}

BANDS_KEYS = {
    "&BANDS": [
        "prefix",
        "outdir",
        "filband",
        "spin_component",
        "lsigma",
        "lp",
        "filp",
        "lsym",
        "no_overlap",
        "plot_2d",
        "firstk",
        "lastk"
    ]
}

BAND_INTERPOLATION_KEYS = {
    "INTERPOLATION": [
        "method",
        "miller_max",
        "check_periodicity",
        "p_metric",
        "scale_sphere"
    ]
}

PROJWFC_KEYS = {
    "PROJWFC": [
        "prefix",
        "outdir",
        "ngauss",
        "degauss",
        "emin",
        "emax",
        "deltae",
        "lsym",
        "diag_basis",
        "pawproj",
        "filpdos",
        "filproj",
        "lwrite_overlaps",
        "lbinary_data",
        "kresolveddos",
        "tdosinboxes",
        "n_proj_boxes",
        "irmin",
        "irmax",
        "plotboxes"
    ]
}

MOLECULARPDOS_KEYS = {
    "INPUTMOPDOS": [
        "xmlfile_full",
        "xmlfile_part",
        "i_atmwfc_beg_full",
        "i_atmwfc_end_full",
        "i_atmwfc_beg_part",
        "i_atmwfc_end_part",
        "i_bnd_beg_full",
        "i_bnd_end_full",
        "i_bnd_beg_part",
        "i_bnd_end_part",
        "fileout",
        "ngauss",
        "degauss",
        "emin",
        "emax",
        "deltae",
        "kresolveddos"
    ]
}

IMPORTEXPORT_BINARY_KEYS = {
    "INPUTPP": [
        "prefix",
        "outdir",
        "direction",
        "newoutdir"
    ]
}


OSCDFT_PP_KEYS = {
    "OSCDFT_PP_NAMELIST": [
        "prefix",
        "outdir"
    ]
}

KCW_KEYS = {
    {
    "CONTROL": [
        "prefix",
        "outdir",
        "calculation",
        "kcw_iverbosity",
        "kcw_at_ks",
        "read_unitary_matrix",
        "spread_thr",
        "homo_only",
        "l_vcut",
        "assume_isolated",
        "spin_component",
        "mp1",
        "mp2",
        "mp3",
        "lrpa"
    ],
    "WANNIER": [
        "seedname",
        "num_wann_occ",
        "num_wann_emp",
        "have_empty",
        "has_disentangle",
        "check_ks"
    ],
    "SCREEN": [
        "niter",
        "nmix",
        "tr2",
        "i_orb",
        "eps_inf",
        "check_spread"
    ],
    "HAM": [
        "do_bands",
        "use_ws_distance",
        "write_hr",
        "on_site_only"
    ]
    }
}

CPPP_KEYS = {
    "INPUTPP": [
        "prefix",
        "fileout",
        "output",
        "outdir",
        "lcharge",
        "lforces",
        "ldynamics",
        "lpdb",
        "lrotation",
        "np1",
        "np2",
        "np3",
        "nframes",
        "ndr",
        "atomic_number"
    ]
}

{
    "INPUTPP": [
        "prefix",
        "fileout",
        "output",
        "outdir",
        "lcharge",
        "lforces",
        "ldynamics",
        "lpdb",
        "lrotation",
        "np1",
        "np2",
        "np3",
        "nframes",
        "ndr",
        "atomic_number"
    ]
}