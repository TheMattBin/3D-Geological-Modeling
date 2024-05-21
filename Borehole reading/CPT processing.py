import pandas as pd
import numpy as np
from scipy.io import loadmat
import re
import os
import glob


Path = os.getcwd()
BH_No = os.listdir(Path)
agsfile_list = []
for i in BH_No[:-1]:
    tmp = os.path.join(Path,i)
    tmp_file = glob.glob(tmp + r'\*.ags')
    agsfile_list.extend(tmp_file)

CPTdf = []


for agsfile in agsfile_list:
    RepNo = agsfile.split('\\')

    # delimiters = '",', '"',',"'
    delimiters = '","'
    # regexPattern = '|'.join(map(re.escape, delimiters))

    f = open(agsfile, 'r', encoding='utf-8', errors='ignore') # 'r' = read
    lines = f.readlines()

    agsdata = [re.split(delimiters, adata) for adata in lines] # split string according to ","

    # Remove " and \n in data
    for idx in range(len(agsdata)):
        agsdata[idx] = [adata.replace('"', '').strip().replace('?', '') for adata in agsdata[idx]]

    # Remove empty rows in data
    emptyid = [i for i, adata in enumerate(agsdata) if (len(adata[0]) == 0) and len(adata) == 1]
    for idx in sorted(emptyid, reverse=True):  # Delete empty lines in reverse order
        del agsdata[idx]

    # combine * (heading) rows
    ssid = [i for i, adata in enumerate(agsdata) if ('*' in adata[0]) and '**' not in adata[0]]  # Index of string containing '*' without '**'
    ssid = sorted(ssid, reverse=True)  # Combined and delete heading lines in reverse order
    for idx, sidx in enumerate(ssid[0:len(ssid) - 1]):
        if sidx == ssid[idx + 1] + 1:
            agsdata[ssid[idx + 1]] += agsdata[sidx]
            del agsdata[sidx]

    # Combine <CONT> rows
    contid = [i for i, adata in enumerate(agsdata) if '<CONT>' in adata[0]]  # Index of string containing '<CONT>'
    contid = sorted(contid, reverse=True)  # Combined and delete <CONT> lines in reverse order
    for idx in contid:
        diff = abs(len(agsdata[idx - 1]) - len(agsdata[idx]))
        if len(agsdata[idx - 1]) > len(agsdata[idx]):
            agsdata[idx].extend([''] * diff)
        elif len(agsdata[idx - 1]) < len(agsdata[idx]):
            agsdata[idx - 1].extend([''] * diff)
        adata = list(map(str.__add__, agsdata[idx - 1], agsdata[idx]))
        for idx2, ad in enumerate(adata):
            adata[idx2] = ad.replace('<CONT>', '')
        agsdata[idx - 1] = adata
        del agsdata[idx]

    # Remove <UNITS> rows
    unitid = [i for i, adata in enumerate(agsdata) if '<UNITS>' in adata[0]]  # Index of string containing '<UNITS>'
    for idx in sorted(unitid, reverse=True):  # Delete <UNITS> rows in reverse order
        del agsdata[idx]

    # Remove ',' in keywords
    for data in agsdata:
        for i, w in enumerate(data):
            if ',' in w and '*' in w:
                data[i] = w.replace(',', '')

    # Conditional check if key words in agsdata to filter lab work or separate AGS files
    agsdata_check = []
    for i in agsdata:
        agsdata_check.extend(i)


    if '**HOLE' in agsdata_check and "**STCN" in agsdata_check:
        # Extract Hole Information (**HOLE)
        dsid = [i for i, t in enumerate(agsdata) if '**' in t[0]] # All indices of the rows that the first elements contain '**'
        dsid.append(len(agsdata))  # Append last index to avoid some info in last paragraph
        Hsid = [i for i, t in enumerate(agsdata) if '**HOLE' in t[0]][0] # First index of string containing '**HOLE'
        Heid = dsid[dsid.index(Hsid) + 1]
        Hdata = agsdata[Hsid:Heid]

        # set the keywords of Hole information to be extracted
        # Hole ID, type, E-coordinate, N-coordinate, Ground level, Final depth, Orientation and Inclination
        kw_Hole = []
        kw_Hole_l = ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP']
        for i in kw_Hole_l:
            if i in Hdata[1]:
                kw_Hole.append(i)

        for hd in Hdata[2:]:
            if len(hd) != len(Hdata[1]):
                hd.extend([''] * abs(len(hd) - len(Hdata[1])))

        Hdict = dict()
        for kwi in kw_Hole:
            kws = kwi.replace('*', '') # heading to create dictionary
            kw_cid = [i for i, hd in enumerate(Hdata[1]) if kwi in hd][0] # First index that contains the keyword of kwi
            kw_info = [hd[kw_cid] for hd in Hdata[2:len(Hdata)]]
            Hdict[kws] = kw_info
        Hinfo = pd.DataFrame(Hdict)
        Hinfo['Report No'] = [RepNo[-2]]*len(Hinfo) #Create column for Report No.

        # Extract shear box testing (general) Information
        SBGsid = [i for i, t in enumerate(agsdata) if "**STCN" in t[0]][0]  # First index of string containing '**GEOL'
        SBGeid = dsid[dsid.index(SBGsid) + 1]  # Last index of string containing '**GEOL'
        SBGdata = agsdata[SBGsid:SBGeid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_SBG = []
        kw_SBG_l = ["*HOLE_ID","*STCN_DPTH","*STCN_RES","*STCN_FRES","*STCN_PWP1","*STCN_PWP2","*STCN_PWP3","*STCN_FRR","*STCN_REF","*STCN_TYP"]
        for i in kw_SBG_l:
            if i in SBGdata[1]:
                kw_SBG.append(i)

        for gd in SBGdata[2:]:
            if len(gd) != len(SBGdata[1]):
                gd.extend([''] * abs(len(gd) - len(SBGdata[1])))

        SBGdict = dict()
        for kwi in kw_SBG:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, gd in enumerate(SBGdata[1]) if kwi in gd][0]  # First index that contains the keyword of kwi
            kw_info = [gd[kw_cid] for gd in SBGdata[2:len(SBGdata)]]
            SBGdict[kws] = kw_info
        SBGinfo = pd.DataFrame(SBGdict)

        # Borehole Info joined with Geology Info
        sub_df_SBG = Hinfo.join(SBGinfo.set_index('HOLE_ID'), on='HOLE_ID')
        CPTdf.append(sub_df_SBG)


if len(CPTdf) != 0:
    dftest = pd.concat(CPTdf, sort=False)
    dftest.to_csv('CPTcomb.csv', index=None)

f.close()