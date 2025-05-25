import os
import re
import glob
import pandas as pd
from typing import List, Dict, Any, Optional

class AGSProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def find_ags_files(self) -> List[str]:
        agsfile_list = []
        for entry in os.listdir(self.base_path):
            entry_path = os.path.join(self.base_path, entry)
            if os.path.isdir(entry_path):
                agsfile_list.extend(glob.glob(os.path.join(entry_path, '*.ags')))
        return agsfile_list

    def clean_ags_lines(self, lines: List[str]) -> List[List[str]]:
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

    def extract_section(self, agsdata: List[List[str]], section: str) -> List[List[str]]:
        dsid = [i for i, t in enumerate(agsdata) if '**' in t[0]]
        dsid.append(len(agsdata))
        sid = [i for i, t in enumerate(agsdata) if section in t[0]][0]
        eid = dsid[dsid.index(sid) + 1]
        return agsdata[sid:eid]

    def extract_keywords(self, data: List[List[str]], keywords: List[str]) -> List[str]:
        return [k for k in keywords if k in data[1]]

    def build_dict(self, data: List[List[str]], keywords: List[str]) -> Dict[str, List[Any]]:
        d = {}
        for kwi in keywords:
            kws = kwi.replace('*', '')
            kw_cid = [i for i, hd in enumerate(data[1]) if kwi in hd][0]
            kw_info = [hd[kw_cid] for hd in data[2:len(data)]]
            d[kws] = kw_info
        return d

    def process(
        self,
        section_configs: List[Dict[str, Any]],
        output_files: Optional[List[str]] = None
    ) -> None:
        agsfile_list = self.find_ags_files()
        results = [[] for _ in section_configs]
        for agsfile in agsfile_list:
            RepNo = agsfile.split(os.sep)
            with open(agsfile, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            agsdata = self.clean_ags_lines(lines)
            agsdata_check = [item for sublist in agsdata for item in sublist]
            for idx, config in enumerate(section_configs):
                if all(sec in agsdata_check for sec in config['required_sections']):
                    dfs = []
                    for sec, kw_list in zip(config['sections'], config['keywords']):
                        data = self.extract_section(agsdata, sec)
                        kws = self.extract_keywords(data, kw_list)
                        dct = self.build_dict(data, kws)
                        dfs.append(pd.DataFrame(dct))
                    # Join all DataFrames on HOLE_ID
                    df = dfs[0]
                    for d in dfs[1:]:
                        df = df.join(d.set_index('HOLE_ID'), on='HOLE_ID')
                    df['Report No'] = [RepNo[-2]] * len(df)
                    results[idx].append(df)
        # Save results
        if output_files:
            for idx, out in enumerate(output_files):
                if results[idx]:
                    pd.concat(results[idx], sort=False).to_csv(out, index=False)

# Example usage for borehole geology and SPT
if __name__ == "__main__":
    processor = AGSProcessor(base_path=os.getcwd())
    section_configs = [
        {   # Geology
            'required_sections': ['**HOLE', '**GEOL'],
            'sections': ['**HOLE', '**GEOL'],
            'keywords': [
                ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP', '*HOLE_ORNT', '*HOLE_INCL'],
                ["*HOLE_ID", "*GEOL_TOP", "*GEOL_BASE", "*GEOL_DESC", "*GEOL_LEG", "*GEOL_GEOL", "*GEOL_GEOL2", "*GEOL_STAT"]
            ]
        },
        {   # SPT
            'required_sections': ['**HOLE', '**ISPT'],
            'sections': ['**HOLE', '**ISPT'],
            'keywords': [
                ['*HOLE_ID', '*HOLE_TYPE', '*HOLE_NATE', '*HOLE_NATN', '*HOLE_GL', '*HOLE_FDEP', '*HOLE_ORNT', '*HOLE_INCL'],
                ["*HOLE_ID", "*ISPT_TOP", "*ISPT_NVAL", "*ISPT_NPEN", "*ISPT_SEAT", "*ISPT_MAIN", "*ISPT_CAS", "*ISPT_WAT", "*ISPT_TYPE", "*ISPT_REM", "*ISPT_INC1", "*ISPT_INC2", "*ISPT_INC3", "*ISPT_INC4", "*ISPT_INC5", "*ISPT_INC6", "*ISPT_PEN1", "*ISPT_PEN2", "*ISPT_PEN3", "*ISPT_PEN4", "*ISPT_PEN5", "*ISPT_PEN6", "*ISPT_LAST"]
            ]
        }
    ]
    output_files = ['GEOLcomb.csv', 'SPTcomb.csv']
    processor.process(section_configs, output_files)