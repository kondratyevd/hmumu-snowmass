import os, sys
import time
import subprocess
from string import Template
import glob
from dask.distributed import Client
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.interpolate import interp1d

style = hep.style.CMS
# style["mathtext.fontset"] = "cm"
# style["mathtext.default"] = "rm"
plt.style.use(style)

from configuration import (
    yield_scales,
    root_files,
    templates,
    autoMCstats,
)

LUMI_RUN2 = 137
TEV_OPTIONS = [13, 14]

COMBINE_OPTIONS = " -L lib/HMuMuRooPdfs_cc.so --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams"

COMBINE_OPTIONS_KAPPA_MU = "-L lib/HMuMuRooPdfs_cc.so --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams --robustFit=1"


def get_significance(args):
    tev = args.pop("tev", 13)
    lumi = args.pop("lumi", 137)
    scenario = args.pop("scenario", "S0")
    pdf_index = args.pop("pdf_index", -1)
    out_path = args.pop("out_path", "datacards/")
    poi = args.pop("poi", "Significance")
    out_name_prefix = args.pop("out_name_prefix", "cms_hmm_combination")
    channel = args.pop("channel", None)

    if pdf_index >= 0:
        scenario = f"{scenario}_pdf{pdf_index}"

    try:
        os.mkdir(out_path)
    except:
        pass

    if tev not in TEV_OPTIONS:
        raise Exception(
            f"Incorrect energy: {tev}. Should be one of the following: {TEV_OPTIONS}"
        )

    if (poi != "Significance") & (channel != "inclusive"):
        return {}
    if ("Run_2" in scenario) & (channel != "inclusive"):
        return {}
    if "Run_2" in scenario:
        tev = 13

    lumiscale = round(lumi / LUMI_RUN2, 5)

    # -------------- Make datacard from template --------------
    datacard_template = templates[poi][channel]
    substitutions = {
        "input_file": root_files[channel]["original"],
        "input_file_new": root_files[channel]["new_mass_res"],
        "lumiscale": lumiscale,
        "commentautostats": autoMCstats[scenario],
    }
    substitutions.update(yield_scales[tev])
    if "Run_2" in scenario:
        substitutions["input_file_new"] = substitutions["input_file"]

    with open(datacard_template, "r") as f:
        tmp = f.read()

    custom_text = Template(tmp).substitute(**substitutions)

    # Save datacard
    out_name = f"{out_name_prefix}_{tev}TeV_{lumi}fb_{scenario}_for{poi}.txt"
    if channel == "ggh":
        out_name = out_name.replace(".txt", "_ggh.txt")
    elif channel == "vbf":
        out_name = out_name.replace(".txt", "_vbf.txt")
    out_fullpath = out_path + out_name

    with open(out_fullpath, "w") as f:
        f.write(custom_text)

    print(f"Saved datacard here: {out_fullpath}")

    # -------------- -------------- -------------- --------------

    if poi == "Significance":
        text2workspace_arguments = "-m 125.38 -L lib/HMuMuRooPdfs_cc.so"
        setParameters = ""
        freezeParameters = ""
        combine_arguments = (
            "-M Significance -m 125.38 --expectSignal=1 -t -1" + COMBINE_OPTIONS
        )
    elif poi == "KappaMuUncertainty":
        text2workspace_arguments = "-m 125.38 -L lib/HMuMuRooPdfs_cc.so -P HiggsAnalysis.CombinedLimit.LHCHCGModels:K1 --PO dohmm=true"
        setParameters = "--setParameters kappa_b=1,kappa_W=1,kappa_Z=1,kappa_tau=1,kappa_t=1,kappa_mu=1"
        freezeParameters = (
            "--freezeParameters=kappa_b,kappa_W,kappa_Z,kappa_tau,kappa_t"
        )
        combine_arguments = (
            "-M MultiDimFit -m 125.38  --algo=singles --bypassFrequentistFit -t -1 --redefineSignalPOIs kappa_mu -n Singles_S2_3invab.totalSyst --setParameterRanges kappa_mu=0.,2. "
            + COMBINE_OPTIONS_KAPPA_MU
        )
    elif poi == "SignalStrengthUncertainty":
        text2workspace_arguments = "-m 125.38 -L lib/HMuMuRooPdfs_cc.so"
        setParameters = ""
        freezeParameters = ""
        combine_arguments = (
            "-M MultiDimFit --algo singles --setParameterRanges r=0.2,2 -m 125.38 --expectSignal=1 -t -1 "
            + COMBINE_OPTIONS_KAPPA_MU
        )

    # Option to freeze PDF index in ggH channel
    if pdf_index >= 0:
        if "--setParameters" in setParameters:
            setParameters += f",pdf_index_ggh={pdf_index}"
        else:
            setParameters = f" --setParameters pdf_index_ggh={pdf_index}"
        if "--freezeParameters" in freezeParameters:
            freezeParameters += ",pdf_index_ggh"
        else:
            freezeParameters = "--freezeParameters=pdf_index_ggh"

    # Uncertainty groupings
    group_unc_const = {
        "pBTag": 0.5,
        "pScaleJPileup": 0.5,
        "pLumi": 0.4,
        "sigTheory": 0.5,
        "bkgTheory": 0.5,
    }
    group_unc_func = {
        "pBTagStat": 0.0,
        "pPrefire": 0.0,
        "pEleID": 0.5,
        "pMuonID": 0.5,
        "pScaleJ": 0.5,
        "pScaleJAbs": 0.2,
        "pScaleJRel": 1,
        "pResJ": 0.5,
        # "Others": 0.0,
    }
    theory_unc = [
        "XSecAndNorm*",
    ]

    additional_options = ""

    # Rescaling uncertainties
    if "S2" in scenario:
        for unc, value in group_unc_const.items():
            expr = get_unc_scale(unc, how="group_const", args={"factor": value})
            additional_options += f" {expr} "
        for unc, floor in group_unc_func.items():
            expr = get_unc_scale(
                unc,
                how="group_const",
                args={"factor": max(floor, 1 / np.sqrt(lumiscale))},
            )
            additional_options += f" {expr} "
        for unc in theory_unc:
            expr = get_unc_scale(unc, how="const", args={"factor": 0.5})
            additional_options += f" {expr} "

    # print(scenario, additional_options)

    # Convert datacards to ROOT files for a given uncertainty scenario
    to_workspace = f"text2workspace.py {out_fullpath} {text2workspace_arguments} {additional_options}"

    subprocess.check_output([to_workspace], shell=True)

    command = f"combineTool.py -d {out_fullpath} {combine_arguments} {setParameters} {freezeParameters}"
    # use ROOT file (contains rescaled uncertainties) instead of a datacard
    command = command.replace(".txt", ".root")

    if poi == "Significance":
        value = float(
            subprocess.check_output([f"{command} | grep Significance:"], shell=True)
            .decode("utf-8")
            .replace("Significance: ", "")
        )
    elif poi == "KappaMuUncertainty":
        output = (
            subprocess.check_output([f"{command} | grep 'kappa_mu :'"], shell=True)
            .decode("utf-8")
            .replace("kappa_mu : ", "")
            .replace("(68%)", "")
        )
        values = [v for v in output.split(" ") if len(v) > 0]
        values = [v for v in values if "/" in v][0].split("/")
        values = [float(v) for v in values]
        value = (abs(values[0]) + abs(values[1])) / 2
    elif poi == "SignalStrengthUncertainty":
        output = (
            subprocess.check_output([f"{command} | grep 'r :'"], shell=True)
            .decode("utf-8")
            .replace("r : ", "")
            .replace("(68%)", "")
        )
        values = [v for v in output.split(" ") if len(v) > 0]
        values = [v for v in values if "/" in v][0].split("/")
        values = [float(v) for v in values]
        value = (abs(values[0]) + abs(values[1])) / 2

    ret = {
        "tev": tev,
        "lumi": lumi,
        "poi": poi,
        "channel": channel,
        "scenario": scenario,
        "value": value,
    }
    return ret


def get_unc_scale(name, how="const", args={}):
    # Different ways to rescale uncertainties
    ret = ""
    if how == "lumi":
        # 1/sqrt(L)
        label = name.replace("*", "").replace("_", "")
        expr = f'expr::scale{label}("1/sqrt(@0)",lumiscale[1])'
        ret = f"--X-nuisance-function '{name}' '{expr}'"
    elif how == "lumi_floor":
        # 1/sqrt(L) until floor value is reached
        if "floor" not in args:
            raise Exception
        floor = args["floor"]
        label = name.replace("*", "").replace("_", "")
        expr = f'expr::scale{label}("max({floor},1/sqrt(@0))",lumiscale[1])'
        ret = f"--X-nuisance-function '{name}' '{expr}'"
    elif how == "const":
        # rescale single uncertainty by a constant factor
        if "factor" not in args:
            raise Exception
        factor = args["factor"]
        ret = f"--X-rescale-nuisance '{name}' {factor}"
    elif how == "group_const":
        # rescale group of uncertainties by a constant factor
        if "factor" not in args:
            raise Exception
        factor = args["factor"]
        ret = f"--X-nuisance-group-function '{name}' '{factor}'"
    return ret


def plot(df_input, params={}):
    # Plot significance scan
    name = params.pop("name", "test")
    scenarios = params.pop("scenarios", ["S0"])
    smoothen = params.pop("smoothen", True)
    poi = params.pop("poi", "Significance")
    channels = params.pop("channels", ["inclusive"])
    out_path = params.pop("out_path", "plots/")
    name = poi

    try:
        os.mkdir(out_path)
    except:
        pass

    fig, ax = plt.subplots()
    # fig.set_size_inches(9, 7)
    fig.set_size_inches(9, 9)

    scenario_titles = {
        "S0": "uncertainties as in Run 2 datacards (not rescaled)",
        "S1": "with Run 2 syst. uncert. (S1)",
        "S2": "with HL-LHC syst. uncert. (S2)",
        "S2_pdf0": "S2 with pdf_index=0",
        "S2_pdf1": "S2 with pdf_index=1",
        "S2_pdf2": "S2 with pdf_index=2",
        "Run_2_sensitivity": "Run 2 sensitivity",
    }

    for s in scenarios:
        for c in channels:
            # if s=="S1" and c=="ggh":
            #    continue
            df = df_input.loc[(df_input.scenario == s) & (df_input.channel == c)]
            # print(s, c, df)
            if df.shape[0] == 0:
                continue

            if s in scenario_titles:
                label = scenario_titles[s]
            else:
                label = s

            ccap = {"ggh": "ggH", "vbf": "VBF"}
            if c != "inclusive":
                label += f": {ccap[c]} channel"

            # opts = {"linewidth": 2}
            # opts_marker = {"marker": "s", "linewidth": 0, "markersize": 10}
            opts = {"linewidth": 2, "marker": "o", "markersize": 10}
            if ("2013" in label) or ("Run 1" in label):
                opts = {
                    "marker": "^",
                    "color": "black",
                    "linewidth": 0,
                    "markersize": 10,
                }
            elif "YR" in label:
                opts["marker"] = "^"
                opts["linewidth"] = 0
            else:
                opts["markerfacecolor"] = "none"

            smoothen = False
            if len(df.lumi.values) < 4:
                smoothen = False
            if smoothen:
                smooth = interp1d(
                    df.lumi.values.tolist(), df.value.values.tolist(), kind="cubic"
                )
                xmin = round(df.lumi.min())
                xmax = round(df.lumi.max())
                x = list(range(xmin, xmax))
                ax.plot(x, smooth(x), **{"linewidth": 2})
                ax.plot(df.lumi, df.value, label=label, **opts)
            else:
                ax.plot(df.lumi, df.value, label=label, **opts)

    hep.cms.label(
        llabel="Phase-2 Projection Preliminary",
        rlabel="3000$\mathrm{fb^{-1}}$(14 TeV)",
        fontsize=20,
        loc=2,
    )
    labels = {
        "Significance": "Expected significance",
        "KappaMuUncertainty": r"Total $\kappa_\mu$ uncert.",
        "SignalStrengthUncertainty": r"Total $\sigma/\sigma_{SM}$ uncert.",
    }
    plt.xlabel(r"L [$\mathrm{fb^{-1}}$]")
    plt.ylabel(labels[poi])
    plt.xlim([0, df_input.lumi.max() * 1.1])
    if poi == "Significance":
        # plt.ylim([0, df_input.value.max() * 1.1])
        plt.ylim([0, 15])
        legend_loc = "lower right"

    elif poi == "KappaMuUncertainty":
        plt.ylim([0, 0.35])
        legend_loc = "center right"
    elif poi == "SignalStrengthUncertainty":
        plt.ylim([0, 0.5])
        legend_loc = "center right"

    plt.legend(
        # loc="best",
        loc=legend_loc,
        prop={"size": "x-small"},
    )

    fig.tight_layout()
    out_name = f"{out_path}/{name}"
    fig.savefig(out_name + ".png")
    fig.savefig(out_name + ".pdf")


if __name__ == "__main__":
    tick = time.time()

    redo_df = False
    # redo_df = True
    # filename = "projections.pkl"
    filename = "projections_jan19.pkl"

    if redo_df:
        lumi_options = [100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000]
        # lumi_options = [137, 300]
        # lumi_options = [300, 400, 500, 600, 700]
        tev_options = [14]
        scenarios = ["S1", "S2"]
        # scenarios = ["S1", "S2", "Run_2_sensitivity"]
        # scenarios = ["Run_2_sensitivity"]
        channels = ["inclusive", "ggh", "vbf"]
        # channels = ["inclusive"]
        pois = ["Significance", "KappaMuUncertainty", "SignalStrengthUncertainty"]
        # pois = ["KappaMuUncertainty"]
        # pois = ["Significance"]
        # pois = ["SignalStrengthUncertainty"]
        arglist = []
        for lumi in lumi_options:
            for tev in tev_options:
                for s in scenarios:
                    for c in channels:
                        for poi in pois:
                            argset = {
                                "tev": tev,
                                "lumi": lumi,
                                "scenario": s,
                                "poi": poi,
                                "channel": c,
                            }
                            arglist.append(argset)

        client = Client(
            processes=True,
            n_workers=min(len(arglist), 20),
            threads_per_worker=1,
            memory_limit="1GB",
        )

        futures = client.map(get_significance, arglist)
        results = client.gather(futures)

        df = pd.DataFrame(
            columns=["tev", "lumi", "scenario", "poi", "channel", "value"]
        )
        for i, r in enumerate(results):
            if not r:
                continue
            df = pd.concat([df, pd.DataFrame(r, index=[i])])

        df.to_pickle(filename)
    else:
        df = pd.read_pickle(filename)

    label_2013 = "Snowmass 2013 predictions"
    yr2018_s1 = "Yellow Report 2018"

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "tev": 14,
                    "lumi": 0,
                    "poi": "Significance",
                    "value": 0,
                    "scenario": ["S1", "S2", "S1", "S2", "S1", "S2"],
                    "channel": ["inclusive", "inclusive", "ggh", "ggh", "vbf", "vbf"],
                },
                index=[-100, -101, -102, -103, -104, -105],
            ),
            df,
            pd.DataFrame(
                {
                    "tev": 14,
                    "lumi": [300, 3000],
                    "poi": "Significance",
                    "value": [2.5, 7.9],
                    "scenario": label_2013,
                    "channel": "inclusive",
                },
                index=[100, 101],
            ),
            pd.DataFrame(
                {
                    "tev": 14,
                    "lumi": 3000,
                    "poi": "KappaMuUncertainty",
                    "value": [0.067, 0.05],
                    "scenario": ["YR 2018 - S1", "YR 2018 - S2"],
                    "channel": "inclusive",
                },
                index=[100, 101],
            ),
            pd.DataFrame(
                {
                    "tev": 14,
                    "lumi": 3000,
                    "poi": "SignalStrengthUncertainty",
                    "value": [0.128, 0.096],
                    "scenario": ["YR 2018 - S1", "YR 2018 - S2"],
                    "channel": "inclusive",
                },
                index=[100, 101],
            ),
        ]
    )
    print(df)
    for poi in ["Significance", "KappaMuUncertainty", "SignalStrengthUncertainty"]:
        # for poi in ["SignalStrengthUncertainty"]:
        parameters = {
            # "scenarios": ["S1", "S2", "S2_pdf0", "S2_pdf1", "S2_pdf2", label_2013],
            "scenarios": [
                "S1",
                "S2",
                label_2013,
                "YR 2018 - S1",
                "YR 2018 - S2",
                "Run_2_sensitivity",
            ],
            # "scenarios": ["S1", label_2013, "YR 2018 - S1"],
            # "scenarios": ["S2", label_2013, "YR 2018 - S2"],
            "poi": poi,
            # "channels": ["inclusive"],
            "channels": ["inclusive", "ggh", "vbf"]
            # "channels": ["ggh"]
        }
        plot(df.loc[df.poi == poi], parameters)
    tock = time.time()
    print(f"Completed in {round(tock-tick, 3)} s.")
