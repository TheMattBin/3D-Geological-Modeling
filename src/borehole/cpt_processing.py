import os
import re
import glob
import pandas as pd
from typing import List, Dict, Any

def find_ags_files(base_path: str) -> List[str]:
    agsfile_list = []
    for entry in os.listdir(base_path):
        entry_path = os.path.join(base_path, entry)
        if os.path.isdir(entry_path):
            agsfile_list.extend(glob.glob(os.path.join(entry_path, '*.ags')))
    return agsfile_list

def clean_ags_lines(lines: List[str]) -> List[List[str]]:
    delimiters = '","'
    agsdata = [re.split(delimiters, adata) for adata in lines]
    for idx in range(len(agsdata)):
        agsdata[idx] = [adata.replace('"', '').strip().replace('?', '') for adata in agsdata[idx]]
    emptyid = [i for i, adata in enumerate(agsdata) if (len(adata[0]) == 0) and len(adata) == 1]
    for idx in sorted(emptyid, reverse=True):
        del agsdata[idx]
    ssid = [i for i, adata in enumerate(agsdata) if ('*' in adata[0]) and '**' not in adata[0]]
    ssid = sorted(ssid, reverse=True)
    for idx, sidx in enumerate(ssid[0:len(ssid) - 1]):
        if sidx == ssid[idx + 1] + 1:
            agsdata[ssid[idx + 1]] += agsdata[sidx]
            del agsdata[sidx]
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
    unitid = [i for i, adata in enumerate(agsdata) if '<UNITS>' in adata[0]]
    for idx in sorted(unitid, reverse=True):
        del agsdata[idx]
    for data in agsdata:
        for i, w in enumerate(data):
            if ',' in w and '*' in w:
                data[i] = w.replace(',', '')
    return agsdata

def extract_section(agsdata: List[List[str]], section: str) -> List[List[str]]:
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

def process_cpt_processing(base_path: str, output_csv: str = 'CPTcomb.csv') -> None:
    agsfile_list = find_ags_files(base_path)
    CPTdf = []
    for agsfile in agsfile_list:
        RepNo = agsfile.split(os.sep)
        with open(agsfile, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        agsdata = clean_ags_lines(lines)
        agsdata_check = [item for sublist in agsdata for item in sublist]
        if '**HOLE' in agsdata_check and '**STCN' in agsdata_check:
            Hdata = extract_section(agsdata, '**HOLE')
            kw_Hole_l = ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP']
            kw_Hole = extract_keywords(Hdata, kw_Hole_l)
            Hdict = build_dict(Hdata, kw_Hole)
            Hinfo = pd.DataFrame(Hdict)
            Hinfo['Report No'] = [RepNo[-2]]*len(Hinfo)
            SBGdata = extract_section(agsdata, '**STCN')
            kw_SBG_l = ["*HOLE_ID","*STCN_DPTH","*STCN_RES","*STCN_FRES","*STCN_PWP1","*STCN_PWP2","*STCN_PWP3","*STCN_FRR","*STCN_REF","*STCN_TYP"]
            kw_SBG = extract_keywords(SBGdata, kw_SBG_l)
            SBGdict = build_dict(SBGdata, kw_SBG)
            SBGinfo = pd.DataFrame(SBGdict)
            sub_df_SBG = Hinfo.join(SBGinfo.set_index('HOLE_ID'), on='HOLE_ID')
            CPTdf.append(sub_df_SBG)
    if CPTdf:
        pd.concat(CPTdf, sort=False).to_csv(output_csv, index=None)

def main():
    process_cpt_processing(base_path=os.getcwd())

if __name__ == "__main__":
    main()