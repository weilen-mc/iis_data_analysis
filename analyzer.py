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