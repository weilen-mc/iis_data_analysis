import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import fnmatch
import datetime
import itertools

import pandas as pd

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy import integrate

from os.path import isfile

import re

import copy

import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import utils

class Analyzer:
        
    
    def __init__(self, loader_inst, ureg):
        self.measurements = {}
        self.selection = []
        self.plotting_selection = []
        self.meas_type_dict = {}
        self.axs_labels = {}
        self.axs_limits = [{}]
        self.lo = loader_inst
        if self.lo.raw_measurements:
            self.measurements = self.lo.raw_measurements

        self.ureg = ureg
            
        
    def set_selection(self, selection):
        if selection:
            sel = selection
        else:
            sel = list(self.measurements.keys())
        self.selection = sel
        self.plotting_selection = sel
        return sel

    def set_plotting_selection(self, plotting_selection):
        self.plotting_selection = plotting_selection

    def set_meas_type_dict(self, meas_type_dict):
        self.meas_type_dict = meas_type_dict

    def set_axs_labels(self, axs_labels):
        self.axs_labels = axs_labels
        
    def reset_measurements(self):
        if self.lo.raw_measurements:
            self.measurements = copy.deepcopy(self.lo.raw_measurements)
            return True
        else:
            logging.warning(f"No raw measurement data found")
            return False

    def print_measurements(self, exclude_keys=['data', 'metadata'], **kwargs):
        df = utils._print_dicts(self.measurements, exclude_keys=exclude_keys, **kwargs)
        if type(df) is bool:
            logging.warning(f"No measurements loaded.")
            return False
        return df

    def print_metadata_of_selection(self, sel=None, exclude_keys=['data'], display_data=True):
        if not utils._sanity_check_selection(self):
            return False

        if sel is None:
            if self.selection:
                sel = self.selection
            else:
                sel = list(self.measurements.keys())

        metadata_dict = {}
        for idx in sel:   
            m = self.measurements[idx]
            excl_m = {k:v for k,v in m.items() if k not in exclude_keys}
            df = pd.DataFrame.from_dict(excl_m, orient='index')
            df = df.style.set_caption(f'Measurement: {idx}')
            metadata_dict[idx] = df
            if display_data:
                display(df)

        return metadata_dict

    
    def rm_outlier(self, axis='Voltage', limits=[], exclude_patterns='_CH1'):
        idces_to_delete = []     
        for idx, m in self.measurements.items():

            if exclude_patterns is not None:
                if type(idx) is str and utils._contains_patterns(idx, exclude_patterns):
                    #print(f'skipping: {idx}')
                    continue

            data = m['data']
        
            if axis in data:
                #mask = (data[axis] > limits[0]) & (data[axis] < limits[1])
                mask = data[axis].between(limits[0], limits[1])
                m['data'] = data[mask]
            else:
                m['data'] = data
                

            try:
                t_0 = m['data']['Time'].iloc[0]
            except IndexError:
                logging.warning(f"No data found after removing zero for idx {idx}. Deleting measurement...")
                idces_to_delete.append(idx)

        for idx in idces_to_delete:
            del self.measurements[idx]

        return True
    
    def rm_zero(self, axis='Voltage', thr=1e-4, exclude_patterns='_CH1'): 
        idces_to_delete = []     
        for idx, m in self.measurements.items():

            if exclude_patterns is not None:
                if type(idx) is str and utils._contains_patterns(idx, exclude_patterns):
                    #print(f'skipping: {idx}')
                    continue

            data = m['data']
        
            if axis in data:
                mask = (data[axis].abs() > thr)
                m['data'] = data[mask]
            else:
                m['data'] = data
                

            try:
                t_0 = m['data']['Time'].iloc[0]
            except IndexError:
                logging.warning(f"No data found after removing zero for idx {idx}. Deleting measurement...")
                idces_to_delete.append(idx)

        for idx in idces_to_delete:
            del self.measurements[idx]

        return True
    
    
    def calc_conductance(self, calc_resistance=False, v_app=0, two_channels=False):
        if not utils._sanity_check_selection(self):
            return False

        for idx in self.selection:
            m = self.measurements[idx]
            data = m['data']
            if calc_resistance:
                if two_channels:
                    if 'channel' not in m:
                        logging.warning(f"Channel attribute not found at idx: {idx}")
                        return False
                    ch = m['channel']
                    if ch == 2:

                        idx_ch1 = f'{idx}_CH1'
                        data_ch1 = self.measurements[idx_ch1]['data']
                        v = data['Voltage']
                        v_ch1 = data_ch1['Voltage']
                        v_new = v.where(v.abs()>v_ch1.abs(), v_ch1)
                        r = v_new.abs() / data['Current'].abs()
                        data['Resistance'] = r
                        data_ch1['Resistance'] = r                    
                else:
                    if v_app != 0:
                        v = v_app
                    else:
                        v = data['Voltage']
                    data['Resistance'] = v / data['Current']
            data['Conductance'] = data['Resistance'].rdiv(1)

        return True

    
    def scale_selection(self, x_axis, y_axis, limits, set_G=0, calc_mean=True, scale_method='add', appendix='scaled'):
        selection_scaled = []
        for idx in self.selection:
            m = copy.deepcopy(self.measurements[idx])
            try:
                if calc_mean:
                    init_G = m['data'][(m['data'][x_axis]>limits[0]) & (m['data'][x_axis]<limits[1])][y_axis].mean()
                else:
                    init_G = m['data'][(m['data'][x_axis]>limits[0]) & (m['data'][x_axis]<limits[1])][y_axis].head(1).iloc[0]
            except IndexError:
                print(f'Index {idx}: No {x_axis} data found in the interval: {limits}. Continuing...')
                continue
                
            if scale_method == 'add':
                m['data'][y_axis] = m['data'][y_axis] - init_G + set_G
            elif scale_method == 'mult':
                m['data'][y_axis] = m['data'][y_axis]/init_G * set_G
            else:
                raise NotImplementedError
            #m['data'] = m['data'].apply(scale_fn)
            new_idx = f'{idx}_{appendix}'
            selection_scaled.append(new_idx)
            self.measurements[new_idx] = m
        return selection_scaled



    def split_selection(self, col, split_points, overlap=0, appendix='split', start_value=None, add_labels={}):
        if type(split_points) is not list and type(split_points) is not dict:
            logging.warning(f"split_points must be of type: list or dict.")
            return False

        if type(split_points) is list:
            points = split_points
            split_points = {}
            for s in self.selection:
                split_points[s] = points


        split_selection = []
        for idx in self.selection:

            for i, p in enumerate(split_points[idx]):
                m = copy.deepcopy(self.measurements[idx])
                data = m['data']
                if i == 0:
                    m['data'] = data[data[col]<p]
                else:
                    p_previous = split_points[idx][i-1] - overlap
                    m['data'] = data[(data[col]>=p_previous) & (data[col]<p)]
                    m['timestamp'] = m['timestamp'] + datetime.timedelta(seconds=p_previous)
                try:
                    m['V_start'] = np.round(m['data']['Voltage'].iloc[1], decimals=1)
                    m['V_end'] = np.round(m['data']['Voltage'].iloc[-1], decimals=1)
                except:
                    m['V_start'] = None
                    m['V_end'] = None

                for k,v in add_labels.items():
                    try:
                        m[k] = v[i]
                    except Exception as e:
                        logging.warning(f"no label found for split measurement: {i}")

                new_idx = f'{idx}_{appendix}_{i}'
                split_selection.append(new_idx)   
                self.measurements[new_idx] = m
                if start_value is not None:
                    self.measurements[new_idx] = self.set_start_time_to_value(new_idx, value=start_value)
            else:  
                m = copy.deepcopy(self.measurements[idx])
                data = m['data']
                m['data'] = data[(data[col]>=p-overlap)]
                m['timestamp'] = m['timestamp'] + datetime.timedelta(seconds=p-overlap)
                try:
                    m['V_start'] = np.round(m['data']['Voltage'].iloc[1], decimals=1)
                    m['V_end'] = np.round(m['data']['Voltage'].iloc[-1], decimals=1)
                except:
                    m['V_start'] = None
                    m['V_end'] = None

                for k,v in add_labels.items():
                    try:
                        m[k] = v[i+1]
                    except Exception as e:
                        logging.warning(f"no label found for split measurement: {i+1}")

                new_idx = f'{idx}_{appendix}_{i+1}'
                split_selection.append(new_idx)
                self.measurements[new_idx] = m
                if start_value is not None:
                    self.measurements[new_idx] = self.set_start_time_to_value(new_idx, value=start_value)

        return split_selection
        


    def merge_selection(self, meta_idx=0):
        if not utils._sanity_check_selection(self):
            return False
            
        merged_measurement = copy.deepcopy(self.measurements[self.selection[meta_idx]])
        if 'data' in merged_measurement:
            merged_measurement.pop('data')
            
        max_rep = 0
        for i, idx in enumerate(self.selection):
            data = self.measurements[idx]['data'].copy()
            if 'Repeat' in data:
                curr_max_rep = data['Repeat'].tail(1)
                data['Repeat'] = data['Repeat'] + max_rep
                max_rep = max_rep + curr_max_rep
            else:
                data['Repeat'] = pd.Series([max_rep+i+1 for x in data.index])

            if 'data' in merged_measurement:
                merged_measurement['data'] = merged_measurement['data'].append(data, ignore_index=True)
            else:
                merged_measurement['data'] = data
                
        self.measurements['merged'] = merged_measurement
        
        return True
        

    def set_start_time_to_value(self, idx, value=0):
        if not self.measurements:
            logging.warning(f"No measurements loaded. Aborting...")
            return False

        if idx in self.measurements:
            m = copy.deepcopy(self.measurements[idx])
        else:
            logging.warning(f"idx {idx} not found in measurements. Aborting...")
            return False

        if not ('data' in m and 'Time' in m['data']):
            logging.warning(f"No time data found for idx {idx}. Aborting...")
            return False

        try:
            t_0 = m['data']['Time'].iloc[0]
            m['data']['Time'] = m['data']['Time'] - t_0 + value
        except IndexError:
            logging.warning(f"No time data found for idx {idx}...")

        return m
        
    def set_start_time_to_value_selection(self, value=0):
        if not utils._sanity_check_selection(self):
            return False

        for idx in self.selection:
            m = self.set_start_time_to_value(idx, value)
            if m:
                self.measurements[idx] = m

        return True

    def normalize_selection(self, cols, norm_const_fn, limits=(), limit_col=None, appendix='norm'):
        if not utils._sanity_check_selection(self):
            return False

        norm_selection = []
        for idx in self.selection:
            m = copy.deepcopy(self.measurements[idx])
            data = m['data']
            
            if limit_col is not None and limits:
                data_sel = data[(data[limit_col] >= limits[0]) & (data[limit_col] <= limits[1])]
            else:
                data_sel = data
                
            for c, nc_fn in zip(cols, norm_const_fn):
                if nc_fn == 'max':
                    norm_const = data_sel[c].max()
                elif nc_fn == 'min':
                    norm_const = data_sel[c].min()
                elif (type(nc_fn) is float or type(nc_fn) is int) and limit_col is not None: 
                    norm_const = data_sel[data_sel[limit_col] >= float(nc_fn)][c].iloc[0]
                else:
                    logging.warning(f"Could not interpret normalization constant function: {nc_fn}. Use: min, max, or value of column: limit_col. Setting normalization constant to 1.")
                    norm_const = 1
                
                m['data'][c] = data[c]/norm_const
                
            new_idx = f'{idx}_{appendix}'
            norm_selection.append(new_idx)
            self.measurements[new_idx] = m

        return norm_selection

    def average_selection(self, cols=None, result_name=None, meta_idx=0):
        if not utils._sanity_check_selection(self):
            return False

        m_avg = copy.deepcopy(self.measurements[self.selection[meta_idx]])
        m_avg['data'].reset_index(inplace=True, drop=True)
        for i, idx in enumerate(self.selection):
            if i == 0:
                continue
            m = copy.deepcopy(self.measurements[idx])
            data = m['data']
            data.reset_index(inplace=True, drop=True)
            if cols is None:
                m_avg['data'] = m_avg['data'] + data
            elif type(cols) is list:
                m_avg['data'][cols] = m_avg['data'][cols] + data[cols]
            else:
                logging.warning(f"Cols needs to be of type: list")
                return False
        if cols is None:
            m_avg['data'] = m_avg['data']/len(self.selection)
        elif type(cols) is list:
            m_avg['data'][cols] = m_avg['data'][cols]/len(self.selection)
        else:
            logging.warning(f"Cols needs to be of type: list")
            return False        

        if result_name is None:
            result_name = f'{self.selection[meta_idx]}_avg'
        self.measurements[result_name] = m_avg

        return [result_name]
        
    def moving_avg(self, axes, window=3, appendix='m_avg'):
        selection_mavg = []
        for idx in self.selection:
            m = copy.deepcopy(self.measurements[idx])
            data = m['data']
            m['data'][axes] = data[axes].rolling(window).mean()
            new_idx = f'{idx}_{appendix}'
            selection_mavg.append(new_idx)
            self.measurements[new_idx] = m
        return selection_mavg


    def concat_selection_in_time(self, meta_idx=0, start_time=None, no_gap=False, appendix='shifted', result_name='concat'):
        if not utils._sanity_check_selection(self):
            return False

        last_timestamp = self.measurements[self.selection[meta_idx]]['timestamp']
        concat_measurement = self.measurements[self.selection[meta_idx]].copy()
        if 'data' in concat_measurement:
            concat_measurement.pop('data')
        
        if start_time is None:
            start_time = 0

        previous_end_time = start_time
        shifted_selection = []
        for i, idx in enumerate(self.selection):
            m = copy.deepcopy(self.measurements[idx])
            if 'timestamp' not in m or 'data' not in m:
                logging.warning(f'no timestamp or data found for idx {idx} of selection. Aborting...')
                return False
            timestamp = m['timestamp']

            if 'Time' not in m['data']:
                logging.warning(f'no time axis found for idx {idx} of selection. Aborting...')
                return False

            data = m['data'].copy()
            if not no_gap:
                if i == 0:
                    duration = 0
                else:
                    duration = data['Time'].tail(1).iloc[0] - data['Time'].head(1).iloc[0]
                delta_t = timestamp - last_timestamp
                m = self.set_start_time_to_value(idx, value=previous_end_time+delta_t.seconds - duration)
            else:
                m = self.set_start_time_to_value(idx, value=previous_end_time)

            data = m['data'].copy()
            previous_end_time = data['Time'].tail(1).iloc[0]
            last_timestamp = timestamp

            new_idx = f'{idx}_{appendix}'
            shifted_selection.append(new_idx)
            self.measurements[new_idx] = m

            if 'data' in concat_measurement:
                concat_measurement['data'] = concat_measurement['data'].append(data, ignore_index=True)
            else:
                concat_measurement['data'] = data
                
        self.measurements[result_name] = concat_measurement
        
        return shifted_selection
        
    
    def calc_power(self, calc_energy=True, calc_power_avg=True, window=None):
        if not utils._sanity_check_selection(self):
            return False

        for idx in self.selection:
            m = self.measurements[idx]
            data = m['data']
        
            if 'Voltage' not in data or 'Current' not in data:
                logging.warning(f"Voltage or Current not in data of measurement: {idx}")
                continue

            data['Power'] = data['Voltage']*data['Current']

            if calc_energy or calc_power_avg:
                data['Energy'] = integrate.cumtrapz(data['Power'], data['Time'], initial=0)

            if calc_power_avg:
                P_avg = data['Energy'].tail(1)/(data['Time'].tail(1).iloc[0] - data['Time'].head(1).iloc[0])
                P_avg = P_avg.iloc[0]
                data['P_avg'] = pd.Series([P_avg for t in data['Time']])
                if window is not None:
                    data['P_avg_rolling'] = data['Power'].rolling(window).mean()


        return True


    def calc_ctml_from_selection(self, x_axis, thr, R1, eps=1e-1, ctml_device_nr_pattern_spec=None, 
                                 calc_fit=True, S=None, c=None, meta_idx=0, 
                                 gen_point_measurements=True, choose_first_point=False, run_selection=[]):
        
        if not utils._sanity_check_selection(self):
            return False
        
        ctml_measurement = self.measurements[self.selection[meta_idx]].copy()
        
        if S is None:
            S = {1:40, 2:32, 3:24, 4:20, 5:16, 6:12, 7:8, 8:4}
            
        if c is None:
            c = {4:0.96, 8: 0.93, 12:0.9, 16:0.87, 20:0.84, 24:0.82, 32:0.77, 40:0.73, 48:0.7}
        
        ctml_data = pd.DataFrame(columns=['Resistance', x_axis, 'S', 'c', 'R_corr'])
        all_device_nrs = []
        for i, idx in enumerate(self.selection):   
            m = self.measurements[idx]
            data = m['data']
            data = data[data['Repeat'].isin(run_selection)] if run_selection else data
            
            if x_axis in data:
                sel_data = data[data[x_axis].sub(thr).abs() < eps]
                
                if sel_data.empty:
                    logging.warning(f"Data Point not found in data for index {idx}. Aborting...")
                    return False
                
                if choose_first_point:
                    sel_data = sel_data.head(1).iloc[0]
                else:
                    sel_data = sel_data.tail(1).iloc[0]
                    

                x = sel_data[x_axis].astype(float)

            else:
                logging.warning(f"x_axis: {x_axis} not found in data for index {idx}. Aborting...")
                return False
            
            if 'Resistance' in sel_data:
                R = sel_data['Resistance'].astype(float)
            else:
                logging.warning(f"Resistance not found in data for index {idx}! Aborting...")
                return False
            
            ctml_device_nr = i+1
                
            if ctml_device_nr_pattern_spec is not None:
                filename = self.lo.files[idx]['filename']
                device_nr_attr = utils._extract_attr_values(filename, ctml_device_nr_pattern_spec)
                if device_nr_attr and 'device_nr' in device_nr_attr:
                    device_nr_str = device_nr_attr['device_nr']
                    if device_nr_str is not None and device_nr_str.isnumeric():
                        ctml_device_nr = int(device_nr_str)
                    else:
                        logging.warning(f"Extracted ctml device nr: {device_nr_str} is not numeric. Continuing with default numbering...")
                else:
                    logging.warning(f"Could not extract device nr using pattern spec: {ctml_device_nr_pattern_spec}. Continuing with default numbering...")
            
            all_device_nrs.append(ctml_device_nr)
            
            S_i = S[ctml_device_nr]
            c_i = c[S_i]
            R_corr = R/c_i
            device_data = {'Resistance': R, x_axis: x, 'S': S_i, 'c': c_i, 'R_corr': R_corr}
            
            if gen_point_measurements:
                point_data = pd.DataFrame(columns=['Resistance', x_axis, 'S', 'c', 'R_corr'])
                point_data = point_data.append(device_data, ignore_index=True)
                
                point_idx = f'{idx}_ctml'
                point_measurement = m.copy()
                point_measurement['data'] = point_data
                
                self.measurements[point_idx] = point_measurement
            
             
            ctml_data = ctml_data.append(device_data, ignore_index=True)
        
        ctml_measurement['data'] = ctml_data.sort_values(by='S')
        ctml_measurement['device_numbers'] = all_device_nrs
        
        
        if calc_fit:
            ctml_fit_measurement = self.measurements[self.selection[meta_idx]].copy()
            
            linear_model=np.polyfit(ctml_data['S'], ctml_data['R_corr'],1)
            m = linear_model[0] * self.ureg('ohm/um')
            b = linear_model[1] * self.ureg('ohm')

            R_contact = b/2
            R_sh = m*2*np.pi*R1
            L_T = b/(2*m)
            
            ctml_measurement['m'] = m
            ctml_measurement['b'] = b
            ctml_measurement['R_contact'] = R_contact
            ctml_measurement['R_sheet'] = R_sh
            ctml_measurement['L_T'] = L_T
            
            linear_model_fn=np.poly1d(linear_model)
            S_fit=np.arange(0,S[1]+1)
            R_fit = linear_model_fn(S_fit)
            fit_data = pd.DataFrame({'S':S_fit, 'Resistance':R_fit})
            ctml_fit_measurement['data'] = fit_data
            ctml_fit_measurement['device_numbers'] = all_device_nrs
            
            self.measurements['ctml_fit'] = ctml_fit_measurement
        
        
        self.measurements['ctml'] = ctml_measurement
        
        return True
           
    def add_attributes_to_selection(self, attr_map):
        if not utils._sanity_check_selection(self):
            return False
        
        for attr_name, attr_list in attr_map.items():
            if type(attr_list) is not list:
                logging.warning(f'Values for Attribute: {attr_name} needs to be of type: list. Ignoring...')
                continue
            if len(attr_list) != len(self.selection):
                logging.warning(f'Number of Values for Attribute: {attr_name} needs to be the same as selection. Ignoring...')
                continue
            for idx, attr in zip(self.selection, attr_list):
                self.measurements[idx][attr_name] = attr

        return True


    def create_point_measurement(self, axis, limits, attr_selection=[], meta_idx=0, id_attr=None, add_columns={}, x_fit_col=None, y_fit_col=None, appendix='pick', all_points=False, result_name='points', **kwargs):
        if not utils._sanity_check_selection(self):
            return False

        points = pd.DataFrame()
        all_point_measurement = {}
        picked_selection = []

        for i, idx in enumerate(self.selection):   
            m = self.measurements[idx]
            point_measurement = copy.deepcopy(m)
            if all_points and i==meta_idx:
                all_point_measurement = copy.deepcopy(m)
            data = m['data']
            if type(limits) is list and type(limits[0]) is list:
                if len(limits) != len(self.selection):
                    logging.warning(f'Limits need to be the same length as selection. Aborting...')
                    return False
                lims = limits[i]
            else:
                lims = limits

            sel_data = utils._pick_points(data, axis, lims, **kwargs)

            if type(sel_data) is bool:
                logging.warning(f'No data points found for index: {idx}. Continuing...')
                continue

            for attr in attr_selection:
                if attr in m:
                    sel_data[attr] = m[attr]
                else:
                    logging.warning(f'Attribute: {attr} not found for index: {idx}. Continuing...')

            if id_attr is not None and id_attr in m:
                sel_data[id_attr] = m[id_attr]

            new_idx = f'{idx}_{appendix}'
            point_measurement['data'] = sel_data
            self.measurements[new_idx] = point_measurement
            picked_selection.append(new_idx)
            
            if all_points:
                points = points.append(sel_data, ignore_index = True)

        if all_points and add_columns:
            if id_attr is None:
                points = utils._add_data_cols(points, add_columns, merge_on=None)
            elif id_attr in points:
                points = utils._add_data_cols(points, add_columns, merge_on=id_attr)
            else:
                logging.warning(f'ID attribute: {id_attr} not found. No columns added...')
        
        if all_points:
            all_point_measurement['data'] = points
            self.measurements[result_name] = all_point_measurement
        
        if all_points and x_fit_col is not None and y_fit_col is not None:
            if x_fit_col in points and y_fit_col in points: 
                x = points[x_fit_col]
                y = points[y_fit_col]
                theta = np.polyfit(x, y, 1)
                y_fit = theta[1] + theta[0] * x
                
                fit_data = pd.DataFrame({x_fit_col:x, y_fit_col:y_fit})
                
                fit_measurement = copy.deepcopy(all_point_measurement)
                fit_measurement['data'] = fit_data
                fit_measurement['m'] = theta[0]
                fit_measurement['b'] = theta[1]
                self.measurements[f'{result_name}_fit'] = fit_measurement
            else:
                logging.warning(f'fitting data not found... Aborting fit')
            
        return picked_selection


    
    def _consecutive_rows_exceed_threshold(self, df, column, threshold, n, up_thr = None):
        """
        Returns a mask indicating the indices where n consecutive rows exceed the threshold in a specific column.

        Parameters:
        - df: pandas DataFrame
        - column: str, name of the column to check
        - threshold: int or float, threshold value
        - n: int, number of consecutive rows

        Returns:
        - numpy array, a boolean mask where True indicates the indices where the condition is met
        """
        if threshold < 0:
            exceed_count = df[column] < threshold
        else:
            exceed_count = df[column] > threshold
        if up_thr is not None:
            if up_thr < 0:
                exceed_count = exceed_count > up_thr
            else:
                exceed_count = exceed_count < up_thr
        exceed_count = exceed_count.astype(int)
        exceed_count = exceed_count.rolling(n).sum() >= n
        exceed_count = exceed_count.fillna(False).astype(bool)

        transitions = np.diff(np.concatenate(([0], exceed_count.astype(int), [0])))
        start_indices = np.where(transitions == 1)[0]

        # Prepend 'n' ones to each segment
        for idx in start_indices:
            exceed_count[max(0, idx - n + 1):idx + 1] = True

        return exceed_count
    
    
    
    def pick_points_trigger(self, trigger_level, axis='Voltage', n_points_per_pulse=5, n_points_before_pulse_end=5, fill_above_trigger_level_points=None, fill_above_up_thr=None, appendix='trig', active_edge='falling'):
        picked_selection = []
        copy_excl_keys = ['data']
        for idx in self.selection:
            m = self.measurements[idx]
            point_measurement = {k: v for k, v in m.items() if k not in copy_excl_keys}
            #point_measurement = copy.deepcopy(m)
            data = m['data'].reset_index(drop=True)
            #self.measurements[idx]['data'] = data
            if active_edge == 'rising':
                if trigger_level < 0:
                    mask = (data[axis].shift(-1, fill_value=0) <= trigger_level) & (data[axis] > trigger_level)
                else:
                    mask = (data[axis].shift(-1, fill_value=0) >= trigger_level) & (data[axis] < trigger_level)
            else:
                if trigger_level <0:
                    mask = (data[axis].shift(-1, fill_value=0) >= trigger_level) & (data[axis] < trigger_level)
                else:
                    mask = (data[axis].shift(-1, fill_value=0) <= trigger_level) & (data[axis] > trigger_level)
                             
                
            end_points = data[mask]
            end_point_idces = end_points.index - n_points_before_pulse_end
            selected_point_idces = end_point_idces[end_point_idces >= 0]
            for i in range(1,n_points_per_pulse):
                #print(selected_point_idx[0:10])
                selected_point_idces = selected_point_idces.union(end_point_idces-i)
            
            if fill_above_trigger_level_points is not None:
                mask_above_trigger_level = self._consecutive_rows_exceed_threshold(data, axis, trigger_level, fill_above_trigger_level_points, up_thr=fill_above_up_thr)
                points_above_trigger_level = data[mask_above_trigger_level].index
                
                selected_point_idces = selected_point_idces.union(points_above_trigger_level)
                #mask = mask | mask_above_trigger_level
            
            selected_points = data.loc[selected_point_idces]
            
            new_idx = f'{idx}_{appendix}'
            point_measurement['data'] = selected_points
            self.measurements[new_idx] = point_measurement
            picked_selection.append(new_idx)
            
        return picked_selection


    def pick_timestamps_trigger(self, axis, trigger_level, timestamp_axis='Time', timestamp_offset=0, n_points_before_pulse_end=0, n_points_per_pulse=1, active_edge='falling'):
        picked_timestamps = {}
        for idx in self.selection:
            m = self.measurements[idx]
            #point_measurement = copy.deepcopy(m)
            data = m['data']
            mask_rising = (data[axis].shift(-1, fill_value=0) >= trigger_level) & (data[axis] < trigger_level)
            mask_falling = (data[axis].shift(-1, fill_value=0) <= trigger_level) & (data[axis] > trigger_level)
            if active_edge == 'rising':
                mask = mask_rising
            elif active_edge == 'falling':
                mask = mask_falling
            else:
                mask = mask_rising | mask_falling
            end_points = data[mask]
            end_point_idces = end_points.index - n_points_before_pulse_end
            selected_point_idces = end_point_idces
            iter_start = 1*np.sign(n_points_per_pulse)
            for i in range(iter_start,n_points_per_pulse):
                selected_point_idces = selected_point_idces.union(end_point_idces-i)

            picked_timestamps[idx] = list(data[timestamp_axis].loc[selected_point_idces] + timestamp_offset)

        return picked_timestamps
          


    def fit_selection(self, model, x_axis, y_axis, limits, fit_x=None, selection_stds=[], appendix='fit', manual_parameters={}, scale_factor=1, fix_first_point=False, **curve_fit_kwargs):
        fitted_selection = []
        for i, (idx, idx_err) in enumerate(itertools.zip_longest(self.selection, selection_stds)):
            m = copy.deepcopy(self.measurements[idx])
            data = m['data']

            if type(limits[0]) is list and len(limits) == len(self.selection):
                lims = limits[i]
            else:
                lims = limits
            sel_data = data[(data[x_axis]>=lims[0]) & (data[x_axis]<=lims[1])]

            x = sel_data[x_axis]             

            if idx_err is not None:
                err_data = self.measurements[idx_err]['data']
                err_data = err_data[(data[x_axis]>=lims[0]) & (data[x_axis]<=lims[1])]            

            if type(y_axis) is not list:
                y_axis = [y_axis]

            if fit_x is None:
                fit_x = x


            data_result = pd.DataFrame()
            data_result[x_axis] = pd.Series(fit_x)

            for y_ax in y_axis:
                print_str = f'Parameters for axis: {y_ax}'
                if idx in manual_parameters:
                    print_str = print_str + ' (manual)'
                    parameters = manual_parameters[idx]
                    covariance = ''
                else:
                    y = sel_data[y_ax]
                    if idx_err is not None:
                        sigma = err_data[y_ax]
                    else:
                        sigma = None

                    # if fix_first_point:
                    #     if 'bounds' in curve_fit_kwargs:
                    #         bounds = curve_fit_kwargs['bounds']
                    #     else:
                    #         bounds = ([], [])

                    parameters, covariance = curve_fit(model, x, y*scale_factor, sigma=sigma, **curve_fit_kwargs)

                fit_y = model(fit_x, *parameters)


                data_result[y_ax] = pd.Series(fit_y)/scale_factor
                print(print_str)
                for i, p in enumerate(parameters):
                    par_name = f'p{i}_{y_ax}'
                    print(f'{par_name} = {p}')
                    m[par_name] = p
                # print('Covariance:')
                # print(covariance)
                # print()

            new_idx = f'{idx}_{appendix}'
            fitted_selection.append(new_idx)
            data_result.reset_index(drop=True, inplace=True)
            m['data'] = data_result
            self.measurements[new_idx] = m

        return fitted_selection



    def calc_errorbars(self, selection_err):
        selection_means = []
        selection_stds = []
        for same_measurement_ids in selection_err:
            same_measurement_data = [self.measurements[idx]['data'].reset_index(drop=True) for idx in same_measurement_ids]
            aggr_data = pd.concat(same_measurement_data)
            #display(aggr_data.head())
            by_row_index = aggr_data.groupby(aggr_data.index)
            data_means = by_row_index.mean()
            m = copy.deepcopy(self.measurements[same_measurement_ids[0]])
            m['data'] = data_means
            new_idx = ''

            for i, idx in enumerate(same_measurement_ids):
                if i != (len(same_measurement_ids)-1) and type(idx) is str:
                    idx = idx.split('_')[0]
                new_idx = new_idx + f'{idx}_'
            new_idx = new_idx + 'mean'
            selection_means.append(new_idx)
            self.measurements[new_idx] = m

            data_std = by_row_index.std()
            copy_excl_keys = ['data']
            m = {k: v for k, v in self.measurements[same_measurement_ids[0]].items() if k not in copy_excl_keys}
            m['data'] = data_std
            new_idx = ''
            for i, idx in enumerate(same_measurement_ids):
                if i != (len(same_measurement_ids)-1) and type(idx) is str:
                    idx = idx.split('_')[0]
                new_idx = new_idx + f'{idx}_'
            new_idx = new_idx + 'std'
            selection_stds.append(new_idx)
            self.measurements[new_idx] = m
            
        return selection_means, selection_stds


                
    def generate_plot(self, x_data, y_data, ax, title=None, xlabel=None, ylabel=None,
                       plot_type='line', x_scale='linear', y_scale='linear', x_lim=(), y_lim=(),
                       plot_settings=None, grid=True, legend_entry=None, x_nticks=None, y_nticks=None, **kwargs):
        
        
            
        if plot_settings is None:
            plot_settings = {'linewidth': 3}
            
        
        if x_scale == 'log':
            pass
            #x_data = x_data[x_data>0]
            #x_data = x_data.abs()
        if y_scale == 'log':
            y_data = y_data.abs()
            

            
        if plot_type != 'scatter':
            ax.plot(x_data, y_data, label=legend_entry, **plot_settings)
        else:
            ax.scatter(x_data, y_data, label=legend_entry, **plot_settings)

        #plt.close();
            
        ax.grid(grid)

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        if x_nticks is not None:
            utils._set_tick_number(ax, 'x', x_nticks, scale=x_scale)
        if y_nticks is not None:
            utils._set_tick_number(ax, 'y', y_nticks, scale=y_scale) 

            
        ax.xaxis.set_tick_params(labelbottom=True)

        if x_lim:
            ax.set_xlim(x_lim)
            print(f'x-axis limits: {x_lim}')
        if y_lim:
            ax.set_ylim(y_lim)
            print(f'y-axis limits: {y_lim}')
        
        if title is not None:
            ax.set_title(title)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        
        
        return ax
        
        


    def generate_measurement_plot(self, idx, ax, meas_type='IV', title_template=None, legend_template=None, x_scale_fn=None, y_scale_fn=None, add_labels={}, run_selection=[], interleave=1, **kwargs):

        if meas_type not in self.meas_type_dict:
            logging.warning(f"Unknown measurement type. Add type to meas_type_dict")
            return False

        if 'x' not in self.meas_type_dict[meas_type] or 'y' not in self.meas_type_dict[meas_type]:
            logging.warning(f"keys 'x' and 'y' missing in meas_type_dict")
            return False

        x_axis = self.meas_type_dict[meas_type]['x']         
        y_axis = self.meas_type_dict[meas_type]['y']


        if type(x_axis) is not str:
            logging.warning(f"x_axis needs to be a string.")
            return False
        
        if type(y_axis) is str:
            y_axis = [y_axis]
        elif type(y_axis) is not list:
            logging.warning(f"y_axis needs to be a string or a list.")
            return False


        xlabel = x_axis if x_axis not in self.axs_labels else self.axs_labels[x_axis]
        ylabel = y_axis[0] if y_axis[0] not in self.axs_labels else self.axs_labels[y_axis[0]]



        if self.measurements:
            if idx not in self.measurements:
                logging.warning(f"Could not find idx {idx} in loaded measurements.")
                return False
        else:
            logging.warning(f"No measurements loaded")
            return False

        m = self.measurements[idx]

        if 'x_scale' in kwargs:
            m['x_scale'] = kwargs['x_scale']
        if 'y_scale' in kwargs:
            m['y_scale'] = kwargs['y_scale']

        add_labels.update({'meas_type': meas_type, 'idx': idx})
        title = utils._parse_template(m, title_template, additional_labels=add_labels)



        for y_a in y_axis:
            data = m['data']
            data = data[data['Repeat'].isin(run_selection)] if run_selection else data
            if x_axis not in data or y_a not in data:
                logging.warning(f"Did not find x_axis: {x_axis} or y_axis: {y_a} for index {idx}. Continuing...")
                continue
            xy_data = data[[x_axis, y_a]][::interleave]
            x_data = data[x_axis][::interleave] 
            y_data = data[y_a][::interleave]

            if x_scale_fn is not None:
                if callable(x_scale_fn):
                    x_data = xy_data.apply(x_scale_fn, axis=1)
                else:
                    logging.warning(f"Provided x_scale_fn is not callable. Continuing without x_data scaling...")

            if y_scale_fn is not None:
                if callable(y_scale_fn):
                    y_data = xy_data.apply(y_scale_fn, axis=1)
                else:
                    logging.warning(f"Provided y_scale_fn is not callable. Continuing without y_data scaling...")
            
            add_labels.update({'x_axis': x_axis, 'y_axis':y_a})
            legend_entry = utils._parse_template(m, legend_template, additional_labels=add_labels)
        
            
            ret = self.generate_plot(x_data, y_data, ax, title=title, 
                                     xlabel=xlabel, ylabel=ylabel, legend_entry=legend_entry, **kwargs)

            if ret:
                ax = ret

        if 'no_legend' in kwargs:
            nl = kwargs['no_legend']
        else:
            nl = False
        if len(y_axis) > 1 and not nl:
            pass
            #ax.legend()

        return ax



    def generate_merged_selection_plot(self, ax, sub_selection, plot_parameters, default_plot_parameters, i=0, **kwargs):


        if type(sub_selection) is not list:
            # if type(sub_selection) is not int and type(sub_selection) is not str:
            #     logging.warning(f"Index {sub_selection} needs to be an integer or string not type: {type(sub_selection)}. Aborting...")
            #     return False

            add_labels = {'i': i}

            if type(plot_parameters) is not dict:
                logging.warning(f"Plot_parameters list hierarchy level is higher then selection hierarchy level. Aborting...")
                return False

            new_parameters = default_plot_parameters.copy()
            new_parameters.update(plot_parameters)

            return sub_selection, self.generate_measurement_plot(sub_selection, ax, add_labels=add_labels, **new_parameters, **kwargs)


        for i_s, s in enumerate(sub_selection):
            new_plot_parameters = plot_parameters.copy()
            if type(new_plot_parameters) is list:
                new_plot_parameters = new_plot_parameters[i_s]
            elif type(new_plot_parameters) is not dict:
                logging.warning(f"Plot_parameters need to be of type list or dict. Aborting...")
                return False

            ret = self.generate_merged_selection_plot(ax, s, new_plot_parameters, default_plot_parameters, i=i_s, **kwargs)

            if ret:
                meta_idx, ax = ret

            

        return meta_idx, ax

    def plot_selection(self, figsize=None, single_plot=False, single_plot_title=None, num_cols=3, font=None, display_plot=True, plot_parameters=None, default_plot_parameters={}, legend_labels = [], legend_pos=None, min_num_meas=2, **kwargs):
        
        if figsize is None:
            if single_plot:
                figsize = (5,5)
            else:
                figsize = (12,8)
        
        
        if font is None:
            font = {'weight' : 'normal',
                    'size'   : 18}
        
        plt.rc('font', **font)


        if plot_parameters is None:
            plot_parameters = default_plot_parameters


        if self.plotting_selection:
            sel = self.plotting_selection
            if type(sel) is int or type(sel) is str:
                sel = [sel]
        else:
            sel = list(self.measurements.keys())

        if type(plot_parameters) is dict:
            plot_parameters = [plot_parameters for i in range(len(sel))]
        elif type(plot_parameters) is not list:
            logging.warning(f"Plot_parameters need to be of type list or dict. Aborting...")
            return False

        if len(plot_parameters) != len(sel):
            logging.warning(f"Plot_parameters and plotting_selection do not have the same length. Aborting...")
            return False
                

        figures = {}
        axs = {}
        if single_plot:
            num_meas = len(sel)
            num_rows = int(np.ceil(num_meas/num_cols))
            figsize = (num_cols*figsize[0], num_rows*figsize[1])
            if 'sharex' in kwargs:
                sx = kwargs['sharex']
            else:
                sx = True
            fig, axs_single_plot = plt.subplots(num_rows, num_cols, figsize=figsize, constrained_layout=True, sharex=sx)
            if single_plot_title is not None and type(single_plot_title) is str:
                m_title = None
                if len(sel)>0:
                    if type(sel[0]) is list:
                        if sel[0][0] in self.measurements:
                            m_title = self.measurements[sel[0][0]]
                    elif sel[0] in self.measurements:
                        m_title = self.measurements[sel[0]]
                if m_title is not None:
                    sp_title = utils._parse_template(m_title, single_plot_title)
                else:
                    sp_title = single_plot_title
                if 'pos_suptitle' in kwargs:
                    pos_suptitle = kwargs['pos_suptitle']
                else:
                    pos_suptitle = 1
                fig.suptitle(sp_title, y=pos_suptitle)
            figures['single_plot'] = fig
            axs['single_plot'] = axs_single_plot      
            
        for i, (s, pp) in enumerate(zip(sel, plot_parameters)):   

            if single_plot:
                ax = axs_single_plot[utils._axs_idx_map(i, num_cols, num_rows, identity=False)]
            else:
                fig, ax = plt.subplots(1, figsize=figsize)

            if 'alpha' in kwargs:
                fig.patch.set_alpha(kwargs['alpha'])

            ret = self.generate_merged_selection_plot(ax, s, pp, default_plot_parameters, **kwargs)


            if ret:
                meta_idx, ax = ret
            
            if 'no_legend' in kwargs:
                nl = kwargs['no_legend']
            else:
                nl = False
 
            if type(s) is list and len(s)>=min_num_meas and not nl:
                if 'markerscale' in kwargs:
                    ms = kwargs['markerscale']
                else:
                    ms = 1
                if 'legend_fontsize' in kwargs:
                    lf = kwargs['legend_fontsize']
                else:
                    lf = None
                if legend_labels:
                    if legend_pos is None:
                        ax.legend(legend_labels, bbox_to_anchor=(1,1), loc="upper left", markerscale=ms, fontsize=lf)
                    elif type(legend_pos) is list:
                        if legend_pos[i]:
                            ax.legend(legend_labels, loc=legend_pos[i], markerscale=ms, fontsize=lf)
                    else:
                        ax.legend(legend_labels, loc=legend_pos, markerscale=ms, fontsize=lf)
                else:
                    if legend_pos is None:
                        ax.legend(bbox_to_anchor=(1,1), loc="upper left", markerscale=ms, fontsize=lf)
                    elif type(legend_pos) is list:
                        if legend_pos[i]:
                            ax.legend(loc=legend_pos[i], markerscale=ms, fontsize=lf)
                    else:
                        ax.legend(loc=legend_pos, markerscale=ms, fontsize=lf)

            fig.tight_layout()

            if not single_plot:
                figures[meta_idx] = fig
                axs[meta_idx] = ax
            
            if not single_plot and display_plot:
                pass
                display(fig)
        

        if single_plot and display_plot:
            pass
            display(fig)       
        
        return figures, axs


    
    def safe_figures(self, dir_path, figures, axs, safename_template=None, subfolder_name='clean', dpi=300):
        
        check = True
        for (idx, f), (_, ax) in zip(figures.items(), axs.items()):

            if idx == 'single_plot':
                sup_t = f._suptitle
                if sup_t is not None:
                    title_str = sup_t.get_text()
                else:
                    title_str = 'Single Plot'
            else:
                title_str = ax.title.get_text()
            
            title = utils._clean_string(title_str)
            
            name = title

            if idx != 'single_plot':
                m = self.measurements[idx]
                if 'x_scale' in m:
                    x_scale = m['x_scale']
                    if x_scale == 'log':
                        name = name + '_xlog'
                if 'y_scale' in m:
                    y_scale = m['y_scale']
                    if y_scale == 'log':
                        name = name + '_ylog'
            
            if type(ax) is np.ndarray:
                for i,a in enumerate(ax):
                    #print(a.get_xscale())
                    if a.get_xscale() == 'log':
                        name = name + f'_x{i}log'
                    if a.get_yscale() == 'log':
                        name = name + f'_y{i}log'
                xlim = ax[0].get_xlim()
            else:
                xlim = ax.get_xlim()

            name = name + f'_x={xlim[0]:.0E}_to_{xlim[1]:.0E}'

            safepath = utils._generate_safe_path(dir_path, subfolder_name, name)
            print(safepath)
            if isfile(str(safepath)+'.png') and check:
                print('Should file be overwritten: yes[y] / no [n] / no to all [nn] / yes to all [yy] / rename [r]:')
                inp = input()
                if inp == 'n':
                    print('Skipping file...')
                    continue
                elif inp == 'y':
                    pass
                elif inp == 'yy':
                    check = False
                elif inp == 'nn':
                    print('Aborting...')
                    break
                elif inp == 'r':
                    safepath = utils._rename_safepath(dir_path, subfolder_name, name)
                else:
                    print('Unknown input. Aborting...')
                    break

            f.savefig(safepath, dpi=dpi, bbox_inches="tight")

            
#     def safe_metadata(self, dir_path, data, subfolder_name='clean'):
#         check = True
#         for idx, d in data:
#             name = f'{idx}_metadata.csv'
#             safepath = self._generate_safe_path(dir_path, subfolder_name, name)
#             print(safepath)
#             with open(safepath, 'x') as csv_file:
#                 csv_writer = csv.writer(csv_file, delimiter=',')
#                 header_row = [f'm [{format(m.units, ".4g")}]', f'b [{format(b.units, ".4g")}]', f'R_contact [{format(R_contact.units, ".4g")}]', f'R_sh [{format(R_sh.units, ".4g")}]', f'L_T [{format(L_T.units, ".4g")}]']
#                 csv_writer.writerow(header_row)
#                 data_row = [m.m, b.m, R_contact.m, R_sh.m, L_T.m]
#                 csv_writer.writerow(data_row)