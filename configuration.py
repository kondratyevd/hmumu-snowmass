yield_scales = {
    13: {
        "ggH_hmm_ggH": 1.0,
        "ggH_hmm_VBF": 1.0,
        "qqH_hmm_ggH": 1.0,
        "qqH_hmm_VBF": 1.0,
        "DYJ01": 1.0,
        "DYJ2": 1.0,
        "EWKZ": 1.0,
        "Top": 1.0,
        "bkg": 1.0,
    },
    14: {
        "ggH_hmm_ggH": 1.246,
        "ggH_hmm_VBF": 1.238,
        "qqH_hmm_ggH": 1.246,
        "qqH_hmm_VBF": 1.238,
        "DYJ01": 1.242,
        "DYJ2": 1.242,
        "EWKZ": 1.296,
        "Top": 1.421,
        "bkg": 1.256,
    },
}

# 14 TeV xsec effect ONLY, nothing else changed
# yield_scales_13tev = {
#    "ggH_hmm_ggH": 1.12,
#    "ggH_hmm_VBF": 1.12,
#    "qqH_hmm_ggH": 1.12,
#    "qqH_hmm_VBF": 1.12,
#    "DYJ01": 1.08,
#    "DYJ2": 1.08,
#    "EWKZ": 1.1,
#    "Top": 1.21,
#    "bkg": 1.092,
# }

# exclusive ggH and VBF channels (effect of jet cut)
# yield_scales_13tev = {
#    "ggH_hmm_ggH": 1.2504,
#    "ggH_hmm_VBF": 1.0939,
#    "qqH_hmm_ggH": 1.6220,
#    "qqH_hmm_VBF": 0.6881,
#    "DYJ01": 1.922,
#    "DYJ2": 1.922,
#    "EWKZ": 1.291,
#    "Top": 1.5545,
#    "bkg": 1.248,
# }


templates = {
    "Significance": {
        "inclusive": "templates/cms_hmm_combination_template.txt",
        "ggh": "templates/check_2020_ggh.txt",
        "vbf": "templates/check_2020_vbf.txt",
    },
    "KappaMuUncertainty": {
        "inclusive": "templates/cms_hmm_combination_template_forKappaMu.txt",
    },
    "SignalStrengthUncertainty": {
        "inclusive": "templates/cms_hmm_combination_template_forKappaMu.txt",
    },
}


# Inputs for combine - original models and with Phase-2 resolution
root_files = {
    "inclusive": {
        "original": "cms_hmm.inputs125.38.root",
        "new_mass_res": "cms_hmm.inputs125.38_new.root",
    },
    "ggh": {
        "original": "check_2020_ggh125.38.inputs.root",
        "new_mass_res": "check_2020_ggh125.38.inputs_new.root",
    },
    "vbf": {
        "original": "check_2020_vbf125.38.inputs.root",
        "new_mass_res": "check_2020_vbf125.38.inputs.root",
    },
}

autoMCstats = {"S1": "", "S2": "#", "Run_2_sensitivity": ""}
