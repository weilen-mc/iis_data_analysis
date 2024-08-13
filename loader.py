import copy
import json
import pandas as pd
from os import listdir, rename
from os.path import isfile
from pathlib import Path

import logging
logger = logging.getLogger()

import utils

def _fix_json_keys(raw_data, substitute):
        return_data = copy.deepcopy(raw_data)
        
        for k, v in raw_data.items():

            if type(v) is dict:
                new_v = _fix_json_keys(v, substitute)
                return_data[k] = new_v
                
            else:
                new_v = v


            if k in substitute:
                new_k = substitute[k]

                
                return_data[new_k] = new_v
                return_data.pop(k)
    
        return return_data


    
def _parse_json(dir_path, file):
    path = dir_path / Path(file)
    with open(path) as fp:
        raw_data = json.load(fp)
        try:
            raw_data = _fix_json_keys(raw_data, {'SMU ':'SMU'})
            data = raw_data['data']['SMU']
            if type(data['Voltage']) is float or type(data['Voltage']) is int:
                return pd.DataFrame(data, index=[0])

            res = pd.DataFrame(data)
            if 'SMU_CH1' in raw_data['data']:
                data_ch1 =  raw_data['data']['SMU_CH1']
                if type(data_ch1['Voltage']) is float or type(data_ch1['Voltage']) is int:
                    return res
                
                
                res = {'CH2': res, 'CH1': pd.DataFrame(data_ch1)}
        except Exception as e:
            res = _parse_json_optolab_control(raw_data)
        

        return res
    
def _parse_json_optolab_control(data):
    translation_dict = {'V_ch1': 'Voltage', 'I': 'Current', 'R': 'Resistance', 'time_ch2': 'Time'}

    clean_data = {}
    for k,v in translation_dict.items():
        clean_data[v] = data[k]
    res = pd.DataFrame(clean_data)
    
    return res


    
def _parse_csv(dir_path, file):
    path = dir_path / Path(file)
    with open(path) as fp:

        raw_data = pd.read_csv(fp, names = list(range(0,8)), na_values=[' '])
        header_row_idx = raw_data.index[raw_data.loc[:,0] == 'DataName'][0]

        raw_data = raw_data.loc[header_row_idx:]
        raw_data.columns = raw_data.iloc[0]
        raw_data = raw_data.iloc[1:]

        assert 'DataName' in raw_data

        raw_data = raw_data.drop('DataName', axis=1)
        raw_data = raw_data.rename(columns=
            {
            " V1": "Voltage",
            " I1": "Current",
            " R":  "Resistance",
            " Time": "Time"
            })
        
        raw_data = raw_data.reset_index(drop=True)
        
        if 'Voltage' in raw_data:
            raw_data['Source'] = raw_data['Voltage']
        raw_data = raw_data.astype(float)
        
        return raw_data

def _parse_zahner(dir_path, file):
    path = dir_path / Path(file)
    with open(path) as fp:
        raw_data = pd.read_csv(fp, delim_whitespace=True, na_values=[' '], skiprows=18)
        
        
        if 'Number' in raw_data:
            raw_data = raw_data.drop('Number', axis=1)
        #display(raw_data.head())
        raw_data = raw_data.rename(columns=
            {
            r"Time/s": "Time",
            r"Current/A": "Current",
            r"Potential/V": "Voltage",
            r"Potential": "Voltage",
            r"Impedance/Ohm": "Resistance",
            r"Phase/deg": "Phase",
            })

        raw_data = raw_data.reset_index(drop=True)
        
        #display(raw_data.head())
        
        if 'Voltage' in raw_data:
            raw_data['Source'] = raw_data['Voltage']
        raw_data = raw_data.astype(float)
        
        return raw_data



class Loader:

    def __init__(self):
        self.files = {}
        self.loaded_filenames = []
        self.raw_measurements = {}
        self.dir_path = Path()
        self.accepted_filetypes = ['.json', '.csv', '.txt']
        self.parsers = [_parse_json, _parse_csv, _parse_zahner]


    def load_files(self, dir_path, must_contain=None, attr_pattern_specs=None, escape=True, choose_pattern_nr=0, sort=None):
        self.files = {}
        
        self.dir_path = dir_path
        
        files = listdir(dir_path)        
        
        files = [f for f in files if isfile(dir_path / Path(f)) and f.endswith(tuple(self.accepted_filetypes))] 
                 
        files.sort()
     
        
        filtered_files = utils._extract_strings_containing_patterns(files, must_contain, escape=escape)
        
        if filtered_files is None:
            logging.warning(f"Directory is empty! {dir_path}")
            return None
        
        for idx, f in enumerate(filtered_files):
            self.files[idx] = {'filename': f}
            
            if attr_pattern_specs is None:
                continue
                           
            attr_dict = utils._extract_attr_values(f, attr_pattern_specs, choose_pattern_nr=choose_pattern_nr)
            
            if attr_dict:
                self.files[idx].update(attr_dict) 

            
        
    
    def print_files(self, **kwargs):
        if not utils._print_dicts(self.files, **kwargs):
            logging.warning(f"No files loaded. Execute load_files() first.")
            return False
        

    def rename_files(self, rename_selection):
        if not self.files:
            logging.warning(f"No files loaded. Execute load_files() first.")
            return False 

        if not self.dir_path:
            logging.warning(f"No directory path found. Execute load_files() first.")
            return None

        if type(rename_selection) is not dict:
            logging.warning(f"rename_selection must be a dict.")
            return False 


        for idx, new_name in rename_selection.items():
            if idx not in self.files:
                logging.warning(f"Did not find file with index {idx}. Continuing...")
                continue

            file = self.files[idx]
            filename = file['filename']
            path = self.dir_path / Path(filename)
            new_path = self.dir_path / Path(new_name)

            #print(f'renaming: {path} to {new_path}')

            rename(path, new_path)

        return True


    def load_raw_data(self, **kwargs):
        #self.raw_measurements = {}
        
        if not self.dir_path:
            logging.warning(f"No directory path found. Execute load_files() first.")
            return False
        
        if not self.files:
            logging.warning(f"No files loaded. Execute load_files() first.")
            return False
        
        if len(self.accepted_filetypes) != len(self.parsers):
            logging.warning(f"A parser needs to be registered for every accepted filetype")
            return False
          
        loaded_filenames = []
        for idx, file in self.files.items():
            if file['filename'] not in self.loaded_filenames:
                if idx in self.raw_measurements:
                    new_idx = len(self.loaded_filenames)
                else:
                    new_idx = idx
                self._load_raw_data_file(new_idx, file)
                self.loaded_filenames.append(file['filename'])

        return True


    def _load_raw_data_file(self, idx, file):
        self.raw_measurements[idx] = {}
        for k, v in file.items():
            if k is not None and k != 'filename':
                self.raw_measurements[idx][k] = v
        
        
        
        for file_type, parser in zip(self.accepted_filetypes, self.parsers):
                filename = file['filename']
                #print(filename)
                if filename.endswith(file_type):
                    try:
                        raw_data = parser(self.dir_path, filename)
                    except Exception as e:
                        logging.warning(f"File: {filename} could not be parsed.")
                        del self.raw_measurements[idx]
                        return False
                    break
                    
        else:
            logging.warning(f"Unknown filetype of file: {filename}")
            return False

        if type(raw_data) is dict:
            new_idx = f'{idx}_CH1'
            self.raw_measurements[new_idx] = copy.deepcopy(self.raw_measurements[idx])
            self.raw_measurements[new_idx]['channel'] = 1
            self.raw_measurements[idx]['channel'] = 2
            self.raw_measurements[idx]['data'] = raw_data['CH2']
            self.raw_measurements[new_idx]['data'] = raw_data['CH1']
        else:
            self.raw_measurements[idx]['data'] = raw_data
        return True
