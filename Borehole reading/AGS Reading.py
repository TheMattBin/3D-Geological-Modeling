# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:22:48 2021

@author: Haifeng.Zou, Matthew.Liu
"""

import numpy as np
import pandas as pd
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

Geoldf = []
SPTdf = []

for agsfile in agsfile_list:
    RepNo = agsfile.split('\\')
    print(agsfile)


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
    for idx in sorted(emptyid,reverse=True): # Delete empty lines in reverse order
        del agsdata[idx]

    # combine * (heading) rows
    ssid = [i for i, adata in enumerate(agsdata) if ('*' in adata[0]) and '**' not in adata[0]] # Index of string containing '*' without '**'
    ssid = sorted(ssid, reverse=True) # Combined and delete heading lines in reverse order
    for idx, sidx in enumerate(ssid[0:len(ssid)-1]):
        if sidx == ssid[idx + 1] + 1:
            agsdata[ssid[idx + 1]] += agsdata[sidx]
            del agsdata[sidx]

    # Combine <CONT> rows
    contid = [i for i, adata in enumerate(agsdata) if '<CONT>' in adata[0]] # Index of string containing '<CONT>'
    contid = sorted(contid, reverse=True) # Combined and delete <CONT> lines in reverse order
    for idx in contid:
        diff = abs(len(agsdata[idx - 1])-len(agsdata[idx]))
        if len(agsdata[idx - 1]) > len(agsdata[idx]):
            agsdata[idx].extend(['']*diff)
        elif len(agsdata[idx - 1]) < len(agsdata[idx]):
            agsdata[idx-1].extend(['']*diff)
        adata = list(map(str.__add__, agsdata[idx - 1], agsdata[idx]))
        for idx2, ad in enumerate(adata):
            adata[idx2] = ad.replace('<CONT>', '')
        agsdata[idx - 1] = adata
        del agsdata[idx]

    # Remove <UNITS> rows
    unitid = [i for i, adata in enumerate(agsdata) if '<UNITS>' in adata[0]] # Index of string containing '<UNITS>'
    for idx in sorted(unitid,reverse=True): # Delete <UNITS> rows in reverse order
        del agsdata[idx]

    # Remove ',' in keywords
    for data in agsdata:
        for i, w in enumerate(data):
            if ',' in w and '*' in w:
                data[i] = w.replace(',','')

    # Ensure key words, e.g. Hole, Geol, SPT are in capital letter
    for key in agsdata:
        for j, val in enumerate(key):
            if val.upper() == '**GEOL':
                key[j] = '**GEOL'
            if val.upper() == '**HOLE':
                key[j] = '**HOLE'
            if val.upper() == '**ISPT':
                key[j] = '**ISPT'

    # Conditional check if key words in agsdata to filter lab work or separate AGS files
    agsdata_check = []
    for i in agsdata:
        agsdata_check.extend(i)

    if '**HOLE' in agsdata_check and '**GEOL' in agsdata_check:
        # Extract Hole Information (**HOLE)
        dsid = [i for i, t in enumerate(agsdata) if '**' in t[0]] # All indices of the rows that the first elements contain '**'
        dsid.append(len(agsdata))  # Append last index to avoid some info in last paragraph
        Hsid = [i for i, t in enumerate(agsdata) if '**HOLE' in t[0]][0] # First index of string containing '**HOLE'
        Heid = dsid[dsid.index(Hsid) + 1]
        Hdata = agsdata[Hsid:Heid]

        # set the keywords of Hole information to be extracted
        # Hole ID, type, E-coordinate, N-coordinate, Ground level, Final depth, Orientation and Inclination
        kw_Hole = []
        kw_Hole_l = ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP', '*HOLE_ORNT', '*HOLE_INCL']
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


        # Extract Geology Information
        Gsid = [i for i, t in enumerate(agsdata) if '**GEOL' in t[0]][0] # First index of string containing '**GEOL'
        Geid = dsid[dsid.index(Gsid) + 1] # Last index of string containing '**GEOL'
        Gdata = agsdata[Gsid:Geid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_Geol = []
        kw_Geol_l = ["*HOLE_ID", "*GEOL_TOP", "*GEOL_BASE", "*GEOL_DESC", "*GEOL_LEG", "*GEOL_GEOL", "*GEOL_GEOL2", "*GEOL_STAT"]
        for i in kw_Geol_l:
            if i in Gdata[1]:
                kw_Geol.append(i)

        for gd in Gdata[2:]:
            if len(gd) != len(Gdata[1]):
                gd.extend([''] * abs(len(gd) - len(Gdata[1])))

        Gdict = dict()
        for kwi in kw_Geol:
            kws = kwi.replace('*', '') # heading to create dictionary
            kw_cid = [i for i, gd in enumerate(Gdata[1]) if kwi in gd][0] # First index that contains the keyword of kwi
            kw_info = [gd[kw_cid] for gd in Gdata[2:len(Gdata)]]
            Gdict[kws] = kw_info
        Ginfo = pd.DataFrame(Gdict)

        # Borehole Info joined with Geology Info
        sub_df_geol = Hinfo.join(Ginfo.set_index('HOLE_ID'), on='HOLE_ID')
        Geoldf.append(sub_df_geol)

    if '**HOLE' in agsdata_check and '**ISPT' in agsdata_check:
        # Extract Hole Information (**HOLE)
        dsid = [i for i, t in enumerate(agsdata) if '**' in t[0]]  # All indices of the rows that the first elements contain '**'
        dsid.append(len(agsdata))  # Append last index to avoid some info in last paragraph
        Hsid = [i for i, t in enumerate(agsdata) if '**HOLE' in t[0]][0]  # First index of string containing '**HOLE'
        Heid = dsid[dsid.index(Hsid) + 1]
        Hdata = agsdata[Hsid:Heid]

        # set the keywords of Hole information to be extracted
        # Hole ID, type, E-coordinate, N-coordinate, Ground level, Final depth, Orientation and Inclination
        kw_Hole = []
        kw_Hole_l = ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP', '*HOLE_ORNT',
                     '*HOLE_INCL']
        for i in kw_Hole_l:
            if i in Hdata[1]:
                kw_Hole.append(i)

        for hd in Hdata[2:]:
            if len(hd) != len(Hdata[1]):
                hd.extend([''] * abs(len(hd) - len(Hdata[1])))

        Hdict = dict()
        for kwi in kw_Hole:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, hd in enumerate(Hdata[1]) if kwi in hd][
                0]  # First index that contains the keyword of kwi
            kw_info = [hd[kw_cid] for hd in Hdata[2:len(Hdata)]]
            Hdict[kws] = kw_info
        Hinfo = pd.DataFrame(Hdict)
        Hinfo['Report No'] = [RepNo[-2]] * len(Hinfo)  # Create column for Report No.


        # Extract SPT Information
        SPTsid = [i for i, t in enumerate(agsdata) if '**ISPT' in t[0]][0]  # First index of string containing '**ISPT'
        SPTeid = dsid[dsid.index(SPTsid) + 1]  # Last index of string containing '**ISPT'
        SPTdata = agsdata[SPTsid:SPTeid]
        SPTdata[1] = [data.replace('ISPT_ID', 'HOLE_ID') for data in SPTdata[1]]

        kw_SPT = []
        kw_SPT_l = ["*HOLE_ID", "*ISPT_TOP", "*ISPT_NVAL", "*ISPT_NPEN", "*ISPT_SEAT", "*ISPT_MAIN", "*ISPT_CAS",
                    "*ISPT_WAT", "*ISPT_TYPE", "*ISPT_REM", "*ISPT_INC1", "*ISPT_INC2", "*ISPT_INC3", "*ISPT_INC4", "*ISPT_INC5",
                    "*ISPT_INC6", "*ISPT_PEN1", "*ISPT_PEN2", "*ISPT_PEN3", "*ISPT_PEN4", "*ISPT_PEN5", "*ISPT_PEN6", "*ISPT_LAST"]
        for i in kw_SPT_l:
            if i in SPTdata[1]:
                kw_SPT.append(i)

        # Make sure length of Geology data is the same
        for sd in SPTdata[2:]:
            if len(sd) != len(SPTdata[1]):
                sd.extend([''] * abs(len(sd) - len(SPTdata[1])))

        SPTdict = dict()
        for kwi in kw_SPT:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, sd in enumerate(SPTdata[1]) if kwi in sd][0]  # First index that contains the keyword of kwi
            kw_info = [sd[kw_cid] for sd in SPTdata[2:len(SPTdata)]]
            SPTdict[kws] = kw_info
        SPTinfo = pd.DataFrame(SPTdict)

        # Borehole Info joined with SPT-N
        sub_df_SPT = Hinfo.join(SPTinfo.set_index('HOLE_ID'), on='HOLE_ID')
        SPTdf.append(sub_df_SPT)

    elif ('**HOLE' in agsdata_check) and ('**GEOL' not in agsdata_check):
        # print(agsfile)
        print('ONLY H')
        continue

    elif ('**HOLE' not in agsdata_check) and ('**GEOL' in agsdata_check):
        # print(agsfile)
        print('ONLY G')
        continue

    elif ('**HOLE' not in agsdata_check) and ('**ISPT' in agsdata_check):
        # print(agsfile)
        print('ONLY S')
        continue

    else:
        # print(agsfile)
        print('NO----HOLE')
        continue

if len(Geoldf) != 0:
    dftest = pd.concat(Geoldf, sort=False)
    dftest.to_csv('GEOLcomb.csv', index=None)
    #dftest.to_csv('GEOLcomb.txt', index=None, sep='\t', mode='a')
if len(SPTdf) != 0:
    dftest = pd.concat(SPTdf, sort=False)
    dftest.to_csv('SPTcomb.csv', index=None)
    #dftest.to_csv('SPTcomb.txt', index=None, sep='\t', mode='a')

f.close()
