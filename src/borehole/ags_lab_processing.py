import os
import re
import glob
import pandas as pd
from typing import List, Dict, Any

def find_ags_files(base_path: str) -> List[str]:
    """
    Recursively find all .ags files in subdirectories of base_path.
    """
    agsfile_list = []
    for entry in os.listdir(base_path):
        entry_path = os.path.join(base_path, entry)
        if os.path.isdir(entry_path):
            agsfile_list.extend(glob.glob(os.path.join(entry_path, '*.ags')))
    return agsfile_list

def clean_ags_lines(lines: List[str]) -> List[List[str]]:
    """
    Clean and split AGS file lines into lists of strings.
    """
    delimiters = '","'
    agsdata = [re.split(delimiters, adata) for adata in lines]
    for idx in range(len(agsdata)):
        agsdata[idx] = [adata.replace('"', '').strip().replace('?', '') for adata in agsdata[idx]]
    # Remove empty rows
    emptyid = [i for i, adata in enumerate(agsdata) if (len(adata[0]) == 0) and len(adata) == 1]
    for idx in sorted(emptyid, reverse=True):
        del agsdata[idx]
    # Combine * (heading) rows
    ssid = [i for i, adata in enumerate(agsdata) if ('*' in adata[0]) and '**' not in adata[0]]
    ssid = sorted(ssid, reverse=True)
    for idx, sidx in enumerate(ssid[0:len(ssid) - 1]):
        if sidx == ssid[idx + 1] + 1:
            agsdata[ssid[idx + 1]] += agsdata[sidx]
            del agsdata[sidx]
    # Combine <CONT> rows
    contid = [i for i, adata in enumerate(agsdata) if '<CONT>' in adata[0]]
    contid = sorted(contid, reverse=True)
    for idx in contid:
        diff = abs(len(agsdata[idx - 1]) - len(agsdata[idx]))
        if len(agsdata[idx - 1]) > len(agsdata[idx]):
            agsdata[idx].extend([''] * diff)
        elif len(agsdata[idx - 1]) < len(agsdata[idx]):
            agsdata[idx - 1].extend([''] * diff)
        adata = list(map(str.__add__, agsdata[idx - 1], agsdata[idx]))
        agsdata[idx - 1] = adata
        del agsdata[idx]
    # Remove <UNITS> rows
    unitid = [i for i, adata in enumerate(agsdata) if '<UNITS>' in adata[0]]
    for idx in sorted(unitid, reverse=True):
        del agsdata[idx]
    # Remove ',' in keywords
    for data in agsdata:
        for i, w in enumerate(data):
            if ',' in w and '*' in w:
                data[i] = w.replace(',', '')
    return agsdata

def extract_section(agsdata: List[List[str]], section: str) -> List[List[str]]:
    """
    Extract a section (e.g., '**HOLE') from agsdata.
    """
    dsid = [i for i, t in enumerate(agsdata) if '**' in t[0]]
    dsid.append(len(agsdata))
    sid = [i for i, t in enumerate(agsdata) if section in t[0]][0]
    eid = dsid[dsid.index(sid) + 1]
    return agsdata[sid:eid]

def extract_keywords(data: List[List[str]], keywords: List[str]) -> List[str]:
    return [k for k in keywords if k in data[1]]

def build_dict(data: List[List[str]], keywords: List[str]) -> Dict[str, List[Any]]:
    d = {}
    for kwi in keywords:
        kws = kwi.replace('*', '')
        kw_cid = [i for i, hd in enumerate(data[1]) if kwi in hd][0]
        kw_info = [hd[kw_cid] for hd in data[2:len(data)]]
        d[kws] = kw_info
    return d

def process_ags_lab_files(base_path: str) -> None:
    agsfile_list = find_ags_files(base_path)
    SBdf, TRIdf, CMPdf, CONdf = [], [], [], []
    for agsfile in agsfile_list:
        RepNo = agsfile.split(os.sep)
        with open(agsfile, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        agsdata = clean_ags_lines(lines)
        agsdata_check = [item for sublist in agsdata for item in sublist]
        # Extract shear box testing (general) Information
        if '**HOLE' in agsdata_check and ('**SHBG' in agsdata_check or '**SHBT' in agsdata_check):
            Hdata = extract_section(agsdata, '**HOLE')
            kw_Hole_l = ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP']
            kw_Hole = extract_keywords(Hdata, kw_Hole_l)
            Hdict = build_dict(Hdata, kw_Hole)
            Hinfo = pd.DataFrame(Hdict)
            Hinfo['Report No'] = [RepNo[-2]] * len(Hinfo)
            SBGdata = extract_section(agsdata, '**SHBG')
            kw_SBG_l = ["*HOLE_ID", "*SAMP_TOP", "*SAMP_REF", "*SAMP_TYPE", "*SPEC_REF", "*SPEC_DPTH", "*SHBG_TYPE", "*SHBG_REM", "*SHBG_PCOH", "*SHBG_PHI", "*SHBG_RCOH", "*SHBG_RPHI", "*FILE_FSET"]
            kw_SBG = extract_keywords(SBGdata, kw_SBG_l)
            SBGdict = build_dict(SBGdata, kw_SBG)
            SBGinfo = pd.DataFrame(SBGdict)
            SBTdata = extract_section(agsdata, '**SHBT')
            kw_SBT_l = ["*HOLE_ID", "*SHBT_TESN", "*SHBT_MC", "*SHBT_BDEN", "*SHBT_DDEN", "*SHBT_NORM", "*SHBT_DISP", "*SHBT_PEAK", "*SHBT_RES", "*SHBT_PDIS", "*SHBT_RDIS", "*SHBT_PDEN", "*SHBT_IVR", "*SHBT_MCI", "*SHBT_MCF"]
            kw_SBT = extract_keywords(SBTdata, kw_SBT_l)
            SBTdict = build_dict(SBTdata, kw_SBT)
            SBTinfo = pd.DataFrame(SBTdict)
            sub_df_SBG = Hinfo.join(SBGinfo.set_index('HOLE_ID'), on='HOLE_ID')
            sub_df_SB = sub_df_SBG.join(SBTinfo.set_index('HOLE_ID'), on='HOLE_ID')
            SBdf.append(sub_df_SB)
        # Extract triaxial test Info
        if '**HOLE' in agsdata_check and ("**TRIG" in agsdata_check and "**TRIX" in agsdata_check):
            Hdata = extract_section(agsdata, '**HOLE')
            kw_Hole_l = ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP']
            kw_Hole = extract_keywords(Hdata, kw_Hole_l)
            Hdict = build_dict(Hdata, kw_Hole)
            Hinfo = pd.DataFrame(Hdict)
            Hinfo['Report No'] = [RepNo[-2]] * len(Hinfo)
            TRIGdata = extract_section(agsdata, '**TRIG')
            kw_TRIG_l = ["*HOLE_ID", "*SAMP_TOP", "*SAMP_REF", "*SAMP_TYPE", "*SPEC_REF", "*SPEC_DPTH", "*TRIG_TYPE", "*TRIG_COND", "*TRIG_REM", "*TRIG_CU", "*TRIG_COH", "*TRIG_PHI", "*FILE_FSET"]
            kw_TRIG = extract_keywords(TRIGdata, kw_TRIG_l)
            TRIGdict = build_dict(TRIGdata, kw_TRIG)
            TRIGinfo = pd.DataFrame(TRIGdict)
            TRITdata = extract_section(agsdata, '**TRIX')
            kw_TRIT_l = ["*HOLE_ID", "*TRIX_TESN", "*TRIX_SDIA", "*TRIX_MC", "*TRIX_CELL", "*TRIX_DEVF", "*TRIX_SLEN", "*TRIX_BDEN", "*TRIX_DDEN", "*TRIX_PWPF", "*TRIX_PWPI", "*TRIX_STRN", "*TRIX_MODE"]
            kw_TRIT = extract_keywords(TRITdata, kw_TRIT_l)
            TRITdict = build_dict(TRITdata, kw_TRIT)
            TRITinfo = pd.DataFrame(TRITdict)
            sub_df_TRIG = Hinfo.join(TRIGinfo.set_index('HOLE_ID'), on='HOLE_ID')
            sub_df_TRI = sub_df_TRIG.join(TRITinfo.set_index('HOLE_ID'), on='HOLE_ID')
            TRIdf.append(sub_df_TRI)
        # Extract compaction test Info
        if '**HOLE' in agsdata_check and ("**CMPG" in agsdata_check and "**CMPT" in agsdata_check):
            Hdata = extract_section(agsdata, '**HOLE')
            kw_Hole_l = ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP']
            kw_Hole = extract_keywords(Hdata, kw_Hole_l)
            Hdict = build_dict(Hdata, kw_Hole)
            Hinfo = pd.DataFrame(Hdict)
            Hinfo['Report No'] = [RepNo[-2]] * len(Hinfo)
            CMPGdata = extract_section(agsdata, '**CMPG')
            kw_CMPG_l = ["*HOLE_ID", "*SAMP_TOP", "*SAMP_REF", "*SAMP_TYPE", "*SPEC_REF", "*SPEC_DPTH", "*CMPG_TYPE", "*CMPG_MOLD", "*CMPG_375", "*CMPG_200", "*CMPG_PDEN", "*CMPG_MAXD", "*CMPG_MCOP", "*CMPG_REM", "*CMPG_FSET"]
            kw_CMPG = extract_keywords(CMPGdata, kw_CMPG_l)
            CMPGdict = build_dict(CMPGdata, kw_CMPG)
            CMPGinfo = pd.DataFrame(CMPGdict)
            CMPTdata = extract_section(agsdata, '**CMPT')
            kw_CMPT_l = ["*HOLE_ID", "*CMPT_TESN", "*CMPT_MC", "*CMPT_DDEN"]
            kw_CMPT = extract_keywords(CMPTdata, kw_CMPT_l)
            CMPTdict = build_dict(CMPTdata, kw_CMPT)
            CMPTinfo = pd.DataFrame(CMPTdict)
            sub_df_CMPG = Hinfo.join(CMPGinfo.set_index('HOLE_ID'), on='HOLE_ID')
            sub_df_CMP = sub_df_CMPG.join(CMPTinfo.set_index('HOLE_ID'), on='HOLE_ID')
            CMPdf.append(sub_df_CMP)
        # Extract consolidation test Info
        if '**HOLE' in agsdata_check and ("**CONG" in agsdata_check and "**CONS" in agsdata_check):
            Hdata = extract_section(agsdata, '**HOLE')
            kw_Hole_l = ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP']
            kw_Hole = extract_keywords(Hdata, kw_Hole_l)
            Hdict = build_dict(Hdata, kw_Hole)
            Hinfo = pd.DataFrame(Hdict)
            Hinfo['Report No'] = [RepNo[-2]] * len(Hinfo)
            CONGdata = extract_section(agsdata, '**CONG')
            kw_CONG_l = ["*HOLE_ID", "*SAMP_TOP", "*SAMP_REF", "*SAMP_TYPE", "*SPEC_REF", "*SPEC_DPTH", "*CONG_TYPE", "*CONG_COND", "*CONG_REM", "*CONG_INCM", "*CONG_INCD", "*CONG_DIA", "*CONG_HIGT", "*CONG_MCI", "*CONG_MCF", "*CONG_BDEN", "*CONG_DDEN", "*CONG_PDEN", "*CONG_SATR", "*CONG_SPRS", "*CONG_SATH", "*FILE_FSET"]
            kw_CONG = extract_keywords(CONGdata, kw_CONG_l)
            CONGdict = build_dict(CONGdata, kw_CONG)
            CONGinfo = pd.DataFrame(CONGdict)
            CONSdata = extract_section(agsdata, '**CONS')
            kw_CONS_l = ["*HOLE_ID", "*CONS_INCN", "*CONS_IVR", "*CONS_INCF", "*CONS_INCE", "*CONS_INMV", "*CONS_INCV", "*CONS_INSC"]
            kw_CONS = extract_keywords(CONSdata, kw_CONS_l)
            CONSdict = build_dict(CONSdata, kw_CONS)
            CONSinfo = pd.DataFrame(CONSdict)
            sub_df_CONG = Hinfo.join(CONGinfo.set_index('HOLE_ID'), on='HOLE_ID')
            sub_df_CON = sub_df_CONG.join(CONSinfo.set_index('HOLE_ID'), on='HOLE_ID')
            CONdf.append(sub_df_CON)
    # Example: Save results
    if SBdf:
        pd.concat(SBdf, sort=False).to_csv('shearbox_comb.csv', index=False)
    if TRIdf:
        pd.concat(TRIdf, sort=False).to_csv('triaxial_comb.csv', index=False)
    if CMPdf:
        pd.concat(CMPdf, sort=False).to_csv('compaction_comb.csv', index=False)
    if CONdf:
        pd.concat(CONdf, sort=False).to_csv('consolidation_comb.csv', index=False)

def main():
    process_ags_lab_files(base_path=os.getcwd())

if __name__ == "__main__":
    main()