OPTIONS="-L lib/HMuMuRooPdfs_cc.so --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams"


combine -M MultiDimFit $1 -n .part3E.snapshot -m 125.38 --rMin -1 --rMax 3 --saveWorkspace $OPTIONS
combine -M MultiDimFit higgsCombine.part3E.snapshot.MultiDimFit.mH125.38.root -n .part3E.freezeAll -m 125.38 --rMin -1 --rMax 3 --algo grid --points 30 --freezeParameters allConstrainedNuisances --snapshotName MultiDimFit $OPTIONS
combine -M MultiDimFit $1 -n .part3E -m 125.38 --rMin -1 --rMax 3 --algo grid --points 30 $OPTIONS

plot1DScan.py higgsCombine.part3E.MultiDimFit.mH125.38.root --others 'higgsCombine.part3E.freezeAll.MultiDimFit.mH125.38.root:FreezeAll:2' -o freeze_second_attempt --breakdown Syst,Stat


