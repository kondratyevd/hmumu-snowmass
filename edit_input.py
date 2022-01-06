import ROOT as rt

rt.gSystem.Load("lib/HMuMuRooPdfs_cc.so")
in_filename = "cms_hmm.inputs125.38.root"
out_filename = "cms_hmm.inputs125.38_new.root"

in_file = rt.TFile.Open(in_filename, "READ")

workspaces = {
    "w45": "cat0",
    "w47": "cat1",
    "w49": "cat2",
    "w51": "cat3",
    "w53": "cat4",
}
workspaces_new = []
pdfs_dict = {}
for ws, cat in workspaces.items():
    if ws not in pdfs_dict.keys():
        pdfs_dict[ws] = []
    for sig in ["ggH", "qqH", "ttH", "wH", "zH"]:
        pdfs_dict[ws].append("{0}_{1}_ggh_pdf".format(sig, cat))


for wname, pdfs in pdfs_dict.items():
    w = in_file.Get(wname)
    # w.Print()
    w_new = rt.RooWorkspace(wname, wname)

    for pdf in pdfs:
        # print(w.pdf(pdf).getNorm())#Print()
        # continue
        mh = w.var("mh_ggh")

        sigma_name = pdf.replace("_pdf", "_fsigma")
        sigma_val = w.function(pdf.replace("_pdf", "_fsigma")).getVal() / 1.46
        new_sigma = rt.RooRealVar(
            sigma_name, sigma_name, sigma_val, sigma_val, sigma_val
        )
        # new_sigma.Print()
        # w.function(pdf.replace("_pdf", "_fsigma")).getVal() * 0.5

        mypdf = rt.RooDoubleCBFast(
            pdf,
            pdf,
            mh,
            w.function(pdf.replace("_pdf", "_fpeak")),
            new_sigma,
            w.function(pdf.replace("_pdf", "_spline_aL")),
            w.function(pdf.replace("_pdf", "_spline_nL")),
            w.function(pdf.replace("_pdf", "_spline_aR")),
            w.function(pdf.replace("_pdf", "_spline_nR")),
        )
        # mypdf.Print()
        getattr(w_new, "import")(mypdf)
        getattr(w_new, "import")(w.function(pdf + "_norm"))
        # print("="*30)
    # w_new.Print()
    workspaces_new.append(w_new)

in_file.Close()

out_file = rt.TFile.Open(out_filename, "RECREATE")
for w in workspaces_new:
    w.Write()
out_file.Close()
