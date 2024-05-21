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

SBdf = []
TRIdf = []
CMPdf = []
CONdf = []

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


    if '**HOLE' in agsdata_check and ('**SHBG' or "**SHBT") in agsdata_check:
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
        SBGsid = [i for i, t in enumerate(agsdata) if '**SHBG' in t[0]][0]  # First index of string containing '**GEOL'
        SBGeid = dsid[dsid.index(SBGsid) + 1]  # Last index of string containing '**GEOL'
        SBGdata = agsdata[SBGsid:SBGeid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_SBG = []
        kw_SBG_l = ["*HOLE_ID", "*SAMP_TOP","*SAMP_REF","*SAMP_TYPE","*SPEC_REF","*SPEC_DPTH","*SHBG_TYPE","*SHBG_REM",
                     "*SHBG_PCOH","*SHBG_PHI","*SHBG_RCOH","*SHBG_RPHI","*FILE_FSET"]
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

        # Extract shear box testing (test) Information
        SBTsid = [i for i, t in enumerate(agsdata) if '**SHBT' in t[0]][0]  # First index of string containing '**GEOL'
        SBTeid = dsid[dsid.index(SBTsid) + 1]  # Last index of string containing '**GEOL'
        SBTdata = agsdata[SBTsid:SBTeid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_SBT = []
        kw_SBT_l = ["*HOLE_ID","*SHBT_TESN","*SHBT_MC","*SHBT_BDEN","*SHBT_DDEN","*SHBT_NORM","*SHBT_DISP","*SHBT_PEAK",
                    "*SHBT_RES","*SHBT_PDIS","*SHBT_RDIS","*SHBT_PDEN","*SHBT_IVR","*SHBT_MCI","*SHBT_MCF"]
        for i in kw_SBT_l:
            if i in SBTdata[1]:
                kw_SBT.append(i)

        for gd in SBTdata[2:]:
            if len(gd) != len(SBTdata[1]):
                gd.extend([''] * abs(len(gd) - len(SBTdata[1])))

        SBTdict = dict()
        for kwi in kw_SBT:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, gd in enumerate(SBTdata[1]) if kwi in gd][0]  # First index that contains the keyword of kwi
            kw_info = [gd[kw_cid] for gd in SBTdata[2:len(SBTdata)]]
            SBTdict[kws] = kw_info
        SBTinfo = pd.DataFrame(SBTdict)

        # Borehole Info joined with Geology Info
        sub_df_SBG = Hinfo.join(SBGinfo.set_index('HOLE_ID'), on='HOLE_ID')
        sub_df_SB = sub_df_SBG.join(SBTinfo.set_index('HOLE_ID'), on='HOLE_ID')
        SBdf.append(sub_df_SB)

    # Extract triaxial test Info
    if '**HOLE' in agsdata_check and ("**TRIG" and "**TRIX") in agsdata_check:
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

        # Extract triaxial test (general) Information
        TRIGsid = [i for i, t in enumerate(agsdata) if "**TRIG" in t[0]][0]  # First index of string containing '**GEOL'
        TRIGeid = dsid[dsid.index(TRIGsid) + 1]  # Last index of string containing '**GEOL'
        TRIGdata = agsdata[TRIGsid:TRIGeid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_TRIG = []
        kw_TRIG_l = ["*HOLE_ID","*SAMP_TOP","*SAMP_REF","*SAMP_TYPE","*SPEC_REF","*SPEC_DPTH",
                     "*TRIG_TYPE","*TRIG_COND","*TRIG_REM","*TRIG_CU","*TRIG_COH","*TRIG_PHI","*FILE_FSET"]
        for i in kw_TRIG_l:
            if i in TRIGdata[1]:
                kw_TRIG.append(i)

        for gd in TRIGdata[2:]:
            if len(gd) != len(TRIGdata[1]):
                gd.extend([''] * abs(len(gd) - len(TRIGdata[1])))

        TRIGdict = dict()
        for kwi in kw_TRIG:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, gd in enumerate(TRIGdata[1]) if kwi in gd][0]  # First index that contains the keyword of kwi
            kw_info = [gd[kw_cid] for gd in TRIGdata[2:len(TRIGdata)]]
            TRIGdict[kws] = kw_info
        TRIGinfo = pd.DataFrame(TRIGdict)

        # Extract triaxial test (test) Information
        TRITsid = [i for i, t in enumerate(agsdata) if "**TRIX" in t[0]][0]  # First index of string containing '**GEOL'
        TRITeid = dsid[dsid.index(TRITsid) + 1]  # Last index of string containing '**GEOL'
        TRITdata = agsdata[TRITsid:TRITeid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_TRIT = []
        kw_TRIT_l = ["*HOLE_ID","*TRIX_TESN","*TRIX_SDIA","*TRIX_MC","*TRIX_CELL","*TRIX_DEVF","*TRIX_SLEN","*TRIX_BDEN",
                     "*TRIX_DDEN","*TRIX_PWPF","*TRIX_PWPI","*TRIX_STRN","*TRIX_MODE"]
        for i in kw_TRIT_l:
            if i in TRITdata[1]:
                kw_TRIT.append(i)

        for gd in TRITdata[2:]:
            if len(gd) != len(TRITdata[1]):
                gd.extend([''] * abs(len(gd) - len(TRITdata[1])))

        TRITdict = dict()
        for kwi in kw_TRIT:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, gd in enumerate(TRITdata[1]) if kwi in gd][0]  # First index that contains the keyword of kwi
            kw_info = [gd[kw_cid] for gd in TRITdata[2:len(TRITdata)]]
            TRITdict[kws] = kw_info
        TRITinfo = pd.DataFrame(TRITdict)

        # Borehole Info joined with Geology Info
        sub_df_TRIG = Hinfo.join(TRIGinfo.set_index('HOLE_ID'), on='HOLE_ID')
        sub_df_TRI = sub_df_TRIG.join(TRITinfo.set_index('HOLE_ID'), on='HOLE_ID')
        TRIdf.append(sub_df_TRI)

    # Extract compaction test Info
    if '**HOLE' in agsdata_check and ("**CMPG" and "**CMPT") in agsdata_check:
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

        # Extract compaction test (general) Information
        CMPGsid = [i for i, t in enumerate(agsdata) if "**CMPG" in t[0]][0]  # First index of string containing '**GEOL'
        CMPGeid = dsid[dsid.index(CMPGsid) + 1]  # Last index of string containing '**GEOL'
        CMPGdata = agsdata[CMPGsid:CMPGeid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_CMPG = []
        kw_CMPG_l = ["*HOLE_ID","*SAMP_TOP","*SAMP_REF","*SAMP_TYPE","*SPEC_REF","*SPEC_DPTH","*CMPG_TYPE","*CMPG_MOLD",
                     "*CMPG_375","*CMPG_200","*CMPG_PDEN","*CMPG_MAXD","*CMPG_MCOP","*CMPG_REM","*CMPG_FSET"]
        for i in kw_CMPG_l:
            if i in CMPGdata[1]:
                kw_CMPG.append(i)

        for gd in CMPGdata[2:]:
            if len(gd) != len(CMPGdata[1]):
                gd.extend([''] * abs(len(gd) - len(CMPGdata[1])))

        CMPGdict = dict()
        for kwi in kw_CMPG:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, gd in enumerate(CMPGdata[1]) if kwi in gd][0]  # First index that contains the keyword of kwi
            kw_info = [gd[kw_cid] for gd in CMPGdata[2:len(CMPGdata)]]
            CMPGdict[kws] = kw_info
        CMPGinfo = pd.DataFrame(CMPGdict)

        # Extract compaction test (test) Information
        CMPTsid = [i for i, t in enumerate(agsdata) if "**CMPT" in t[0]][0]  # First index of string containing '**GEOL'
        CMPTeid = dsid[dsid.index(CMPTsid) + 1]  # Last index of string containing '**GEOL'
        CMPTdata = agsdata[CMPTsid:CMPTeid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_CMPT = []
        kw_CMPT_l = ["*HOLE_ID","*CMPT_TESN","*CMPT_MC","*CMPT_DDEN"]
        for i in kw_CMPT_l:
            if i in CMPTdata[1]:
                kw_CMPT.append(i)

        for gd in CMPTdata[2:]:
            if len(gd) != len(CMPTdata[1]):
                gd.extend([''] * abs(len(gd) - len(CMPTdata[1])))

        CMPTdict = dict()
        for kwi in kw_CMPT:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, gd in enumerate(CMPTdata[1]) if kwi in gd][0]  # First index that contains the keyword of kwi
            kw_info = [gd[kw_cid] for gd in CMPTdata[2:len(CMPTdata)]]
            CMPTdict[kws] = kw_info
        CMPTinfo = pd.DataFrame(CMPTdict)

        # Borehole Info joined with Geology Info
        sub_df_CMPG = Hinfo.join(CMPGinfo.set_index('HOLE_ID'), on='HOLE_ID')
        sub_df_CMP = sub_df_CMPG.join(CMPTinfo.set_index('HOLE_ID'), on='HOLE_ID')
        CMPdf.append(sub_df_CMP)

    # Extract consolidation test Info
    if '**HOLE' in agsdata_check and ("**CONG" and "**CONS") in agsdata_check:
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

        # Extract consolidation test (general) Information
        CONGsid = [i for i, t in enumerate(agsdata) if "**CONG" in t[0]][0]  # First index of string containing '**GEOL'
        CONGeid = dsid[dsid.index(CONGsid) + 1]  # Last index of string containing '**GEOL'
        CONGdata = agsdata[CONGsid:CONGeid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_CONG = []
        kw_CONG_l = ["*HOLE_ID","*SAMP_TOP","*SAMP_REF","*SAMP_TYPE","*SPEC_REF","*SPEC_DPTH","*CONG_TYPE","*CONG_COND",
                     "*CONG_REM","*CONG_INCM","*CONG_INCD","*CONG_DIA","*CONG_HIGT","*CONG_MCI","*CONG_MCF","*CONG_BDEN",
                     "*CONG_DDEN","*CONG_PDEN","*CONG_SATR","*CONG_SPRS","*CONG_SATH","*FILE_FSET"]
        for i in kw_CONG_l:
            if i in CONGdata[1]:
                kw_CONG.append(i)

        for gd in CONGdata[2:]:
            if len(gd) != len(CONGdata[1]):
                gd.extend([''] * abs(len(gd) - len(CONGdata[1])))

        CONGdict = dict()
        for kwi in kw_CONG:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, gd in enumerate(CONGdata[1]) if kwi in gd][0]  # First index that contains the keyword of kwi
            kw_info = [gd[kw_cid] for gd in CONGdata[2:len(CONGdata)]]
            CONGdict[kws] = kw_info
        CONGinfo = pd.DataFrame(CONGdict)

        # Extract consolidation test (test) Information
        CONSsid = [i for i, t in enumerate(agsdata) if "**CONS" in t[0]][0]  # First index of string containing '**GEOL'
        CONSeid = dsid[dsid.index(CONSsid) + 1]  # Last index of string containing '**GEOL'
        CONSdata = agsdata[CONSsid:CONSeid]

        # set the keywords of Geology information to be extracted
        # Hole ID, Top, Base, Description, Geology Legend, Geological Code, Formation
        kw_CONS = []
        kw_CONS_l = ["*HOLE_ID","*CONS_INCN","*CONS_IVR","*CONS_INCF","*CONS_INCE","*CONS_INMV","*CONS_INCV","*CONS_INSC"]
        for i in kw_CONS_l:
            if i in CONSdata[1]:
                kw_CONS.append(i)

        for gd in CONSdata[2:]:
            if len(gd) != len(CONSdata[1]):
                gd.extend([''] * abs(len(gd) - len(CONSdata[1])))

        CONSdict = dict()
        for kwi in kw_CONS:
            kws = kwi.replace('*', '')  # heading to create dictionary
            kw_cid = [i for i, gd in enumerate(CONSdata[1]) if kwi in gd][0]  # First index that contains the keyword of kwi
            kw_info = [gd[kw_cid] for gd in CONSdata[2:len(CONSdata)]]
            CONSdict[kws] = kw_info
        CONSinfo = pd.DataFrame(CONSdict)

        # Borehole Info joined with Geology Info
        sub_df_CONG = Hinfo.join(CONGinfo.set_index('HOLE_ID'), on='HOLE_ID')
        sub_df_CON = sub_df_CONG.join(CONSinfo.set_index('HOLE_ID'), on='HOLE_ID')
        CONdf.append(sub_df_CON)

'''
if len(SBdf) != 0:
    dftest = pd.concat(SBdf, sort=False)
    dftest.to_csv('SHBcomb.csv', index=None)
if len(CONdf) != 0:
    dftest = pd.concat(CONdf, sort=False)
    dftest.to_csv('CONcomb.csv', index=None)
if len(CMPdf) != 0:
    dftest = pd.concat(CMPdf, sort=False)
    dftest.to_csv('CMPcomb.csv', index=None)
'''
f.close()