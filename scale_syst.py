import sys
import json

##
inname = sys.argv[1]
outname = sys.argv[2]


inputfile = open(inname, "r")
lines = inputfile.readlines()
inputfile.close()
outputfile = open(outname, "w")
# outputfile.write(lines)

jsonfile = open("syst_grouping.json", "r")
groups_json = json.load(jsonfile)
for i in groups_json.keys():
    group_line = i + " " + "group = "
    print("*" * 40)
    print(i)
    print("*" * 40)
    for isyst in groups_json[i]:
        # if("csv" in isyst):
        print(isyst)
        group_line += isyst + " "
    lines.append(group_line + "\n")


outputfile.writelines(lines)
outputfile.close()

"""
syst_start = False
syst_end = False
for iline in range(len(lines)):
    if("BR_hmm" in lines[iline]):
        syst_start = True
    if(syst_start and not syst_end):
        #if("csv" in lines[iline].split()[0]):
        print(lines[iline].split()[0])
        print(lines[iline].split()[1])
    if("puWeight_2018" in lines[iline]):
        syst_end = True
"""
