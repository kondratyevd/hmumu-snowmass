import os, sys
import time
import subprocess
from string import Template
import glob
from dask.distributed import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.interpolate import interp1d

style = hep.style.CMS
# style["mathtext.fontset"] = "cm"
# style["mathtext.default"] = "rm"
plt.style.use(style)

LUMI_RUN2 = 137
TEV_OPTIONS = [13, 14]

COMBINE_OPTIONS = "-L lib/HMuMuRooPdfs_cc.so --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams"

YIELD_SCALES_13TeV = {
    "ggH_hmm_ggH": 1.0,
    "ggH_hmm_VBF": 1.0,
    "qqH_hmm_ggH": 1.0,
    "qqH_hmm_VBF": 1.0,
    "DYJ01": 1.0,
    "DYJ2": 1.0,
    "EWKZ": 1.0,
    "Top": 1.0,
    "bkg": 1.0,
}

YIELD_SCALES_14TeV = {
    "ggH_hmm_ggH": 1.2504,
    "ggH_hmm_VBF": 1.0939,
    "qqH_hmm_ggH": 1.6220,
    "qqH_hmm_VBF": 0.6881,
    "DYJ01": 1.7626,
    "DYJ2": 1.7626,
    "EWKZ": 1.20,
    "Top": 1.5545,
    "bkg": 1.1533,
}

def get_significance(args):
    tev = args.pop("tev", 13)
    lumi = args.pop("lumi", 137)
    scenario = args.pop("scenario", "S0")
    out_path = args.pop("out_path", "datacards/")
    out_name_prefix = args.pop("out_name_prefix", "cms_hmm_combination")

    try:
        os.mkdir(out_path)
    except:
        pass

    if tev not in TEV_OPTIONS:
        raise Exception(
            f"Incorrect energy: {tev}. Should be one of the following: {TEV_OPTIONS}"
        )

    lumiscale = round(lumi / LUMI_RUN2, 5)
    substitutions = {
        "input_file": "cms_hmm.inputs125.38.root",
        "input_file_new": "cms_hmm.inputs125.38_new.root",
        "lumiscale": lumiscale
    }

    if tev == 13:
        substitutions.update(YIELD_SCALES_13TeV)

    elif tev == 14:
        substitutions.update(YIELD_SCALES_14TeV)

    out_name = f"{out_name_prefix}_{tev}TeV_{lumi}fb_{scenario}.txt"
    out_fullpath = out_path + out_name

    with open("templates/cms_hmm_combination_template.txt", "r") as f:
        tmp = f.read()

    custom_text = Template(tmp).substitute(**substitutions)

    with open(out_fullpath, "w") as f:
        f.write(custom_text)

    print(f"Saved datacard here: {out_fullpath}")

    command = (
        f"combineTool.py -d {out_fullpath} -M Significance -m 125.38 --expectSignal=1 -t -1 "
        + COMBINE_OPTIONS
    )
    # command = f"combineTool.py -d {out_fullpath} -M MultiDimFit -m 125.38 --rMin -1 --rMax 5 " + COMBINE_OPTIONS

    theory_unc = [
        "BR_hmm",
        "EWKZjjPartonShower",
        "SignalPartonShower",
        "LHEFac*",
        "LHERen*",
        "PDF*",
        "QCDscale_*",
        "THU_*",
        "XSecAndNorm*",
        "pdf_*",
    ]

    exp_unc = {
        #
        "fewz_*": 0,
        "CMS_scale_j_*": 0.01,
        "lumi_*": 0.01,
        "CMS_eff_*": 0.005,
        "CMS_lepmva_*": 0.005,
        "CMS_pileup_*": 0,
        "CMS_prefiring_*": 0,
        "CMS_btag_*": 0,
        "CMS_ps": 0,
        "CMS_ps*": 0,
        "CMS_qgl": 0,
        "CMS_res_*": 0,
        # "CMS_scale_*": 0,
        "CMS_trig*": 0,
        "CMS_ue": 0,
        "DYModel": 0,
        "MuScale_*": 0,
        "prefiring_*": 0,
        "puWeight_*": 0,
    }
    additional_options = ""
    if scenario == "S0":
        pass
    elif scenario == "S1":
        # rescale statistical uncertainties by 1/sqrt(L)
        # - uncertainties on background fit parameters
        # expr = get_unc_scale(unc_name, how="lumi")
        pass
    elif scenario == "S2":
        # rescale statistical uncertainties by 1/sqrt(L)
        # - uncertainties on background fit parameters
        # rescale theory uncertainties by 0.5
        # - xSec & normalization
        # - parton shower
        # - QCD scale
        # rescale experimental systematics by max(X, 1/sqrt(L))
        # - signal fit parameters
        # - JES and JER
        # - Lumi
        for unc in theory_unc:
            expr = get_unc_scale(unc, how="const", args={"factor": 0.5})
            additional_options += f" {expr} "
        for unc, floor in exp_unc.items():
            # expr = get_unc_scale(unc, how="lumi_floor", args={"floor": floor})
            expr = get_unc_scale(
                unc, how="const", args={"factor": max(floor, 1 / np.sqrt(lumiscale))}
            )
            additional_options += f" {expr} "

    # print(additional_options)

    to_workspace = f"text2workspace.py {out_fullpath} -L lib/HMuMuRooPdfs_cc.so {additional_options}"
    subprocess.check_output([to_workspace], shell=True)

    command = command.replace(".txt", ".root")

    significance = float(
        subprocess.check_output([f"{command} | grep Significance:"], shell=True)
        .decode("utf-8")
        .replace("Significance: ", "")
    )

    ret = {"tev": tev, "lumi": lumi, "significance": significance, "scenario": scenario}
    return ret


def get_unc_scale(name, how="const", args={}):
    ret = ""
    if how == "lumi":
        label = name.replace("*", "").replace("_", "")
        expr = f'expr::scale{label}("1/sqrt(@0)",lumiscale[1])'
        ret = f"--X-nuisance-function '{name}' '{expr}'"
    elif how == "lumi_floor":
        if "floor" not in args:
            raise Exception
        floor = args["floor"]
        label = name.replace("*", "").replace("_", "")
        expr = f'expr::scale{label}("max({floor},1/sqrt(@0))",lumiscale[1])'
        ret = f"--X-nuisance-function '{name}' '{expr}'"
    elif how == "const":
        if "factor" not in args:
            raise Exception
        factor = args["factor"]
        ret = f"--X-rescale-nuisance '{name}' {factor}"
    return ret


def plot(df_input, params={}):
    name = params.pop("name", "test")
    scenarios = params.pop("scenarios", ["S0"])
    smoothen = params.pop("smoothen", True)
    out_path = params.pop("out_path", "plots/")

    try:
        os.mkdir(out_path)
    except:
        pass

    fig, ax = plt.subplots()

    scenario_titles = {
        "S0": "uncertainties as in Run 2 datacards (not rescaled)",
        "S1": "Uncertainty scenario 1",
        "S2": "Uncertainty scenario 2",
    }

    for s in scenarios:
        df = df_input.loc[df_input.scenario == s]
        if df.shape[0] == 0:
            continue

        if s in scenario_titles:
            label = scenario_titles[s]
        else:
            label = s

        opts = {}
        if "Run 1" in label:
            opts = {"marker": "^", "color": "black", "linewidth": 0, "markersize": 10}

        if len(df.lumi.values) < 4:
            smoothen = False
        if smoothen:
            smooth = interp1d(
                df.lumi.values.tolist(), df.significance.values.tolist(), kind="cubic"
            )
            xmin = round(df.lumi.min())
            xmax = round(df.lumi.max())
            x = list(range(xmin, xmax))
            ax.plot(x, smooth(x), label=label, **opts)
        else:
            ax.plot(df.lumi, df.significance, label=label, **opts)

    hep.cms.label(data=True, paper=False, year="", rlabel="14 TeV")

    plt.xlabel(r"L, $fb^{-1}$")
    plt.ylabel("Expected significance")
    plt.xlim([0, df_input.lumi.max() * 1.1])
    plt.ylim([0, df_input.significance.max() * 1.2])

    plt.legend(
        loc="best",
        prop={"size": "small"},
    )

    fig.tight_layout()
    out_name = f"{out_path}/{name}"
    fig.savefig(out_name + ".png")
    # fig.savefig(out_name + ".pdf")


if __name__ == "__main__":
    tick = time.time()

    redo_df = True
    filename = "projections.pkl"

    if redo_df:
        #lumi_options = [100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000]
        lumi_options = [3000]
        tev_options = [14]
        # scenarios = ["S0", "S1", "S2"]
        scenarios = ["S1", "S2"]
        arglist = []
        for lumi in lumi_options:
            for tev in tev_options:
                for s in scenarios:
                    argset = {"tev": tev, "lumi": lumi, "scenario": s}
                    arglist.append(argset)

        client = Client(
            processes=True,
            n_workers=min(len(arglist), 20),
            threads_per_worker=1,
            memory_limit="1GB",
        )

        futures = client.map(get_significance, arglist)
        results = client.gather(futures)

        df = pd.DataFrame(columns=["tev", "lumi", "significance", "scenario"])
        for i, r in enumerate(results):
            df = pd.concat([df, pd.DataFrame(r, index=[i])])

        df.to_pickle(filename)
    else:
        df = pd.read_pickle(filename)

    label_2013 = "Predictions after Run 1"

    df = pd.concat(
        [
            pd.DataFrame(
                {"tev": 14, "lumi": 0, "significance": 0, "scenario": ["S1", "S2"]},
                index=[-100, -101],
            ),
            df,
            pd.DataFrame(
                {
                    "tev": 14,
                    "lumi": [300, 3000],
                    "significance": [2.5, 7.5],
                    "scenario": label_2013,
                },
                index=[100, 101],
            ),
        ]
    )
    print(df)
    parameters = {
        "scenarios": ["S1", "S2", label_2013],
    }
    plot(df, parameters)
    tock = time.time()
    print(f"Completed in {round(tock-tick, 3)} s.")
