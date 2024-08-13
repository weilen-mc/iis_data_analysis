from os import makedirs
from os.path import isfile
from pathlib import Path
import re
import logging
import numpy as np
import pandas as pd
import matplotlib
logger = logging.getLogger()

def _extract_string_between_two_patterns(string, start, end):
        pattern = start+'(.*?)'+end

        m = re.search(pattern, string)
        if m:
            return m.group(1)
    
    
    
def _extract_strings_containing_patterns(string_list, patterns, escape=True):
    if not string_list:
        return None
    
    if not patterns:
        return string_list
    
    if type(patterns) is str:
        patterns = [patterns]
    elif type(patterns) is not list:
        return None
    
    r_str = ''
    for p in patterns:
        if escape:
            p = p.replace('\\', '')
            p = re.escape(p)
            
        r_str = r_str + f'(?=.*{p})'
    r = re.compile(r_str)

    return list(filter(r.match, string_list))
    

def _contains_patterns(string, patterns):
    if type(patterns) is str:
        patterns = [patterns]
    elif type(patterns) is not list:
        logging.warning(f"Wrong type of patterns format")
        return False
    
    r_str = ''
    for p in patterns:
        r_str = r_str + f'(?=.*{p})'
    r = re.compile(r_str)

    return r.search(string)


def _extract_attr_values(string, attr_pattern_specs, choose_pattern_nr=0):
    # meas_name_patterns = {'timestamp':{'start':['[_', 'jk'], 'end': ['h', '9s'], 'trans_fnct': fnct, 'must_contain': 'csv'},
    #                        'meas_name': 'uuu',
    #                        'device': ['_d', '_D']}
    attr_dict = {}
    for attr, pattern_spec in attr_pattern_specs.items():
        kwargs = None
        trans_fnct = None
        if type(pattern_spec) is not dict:
            logging.warning(f"attr_patter_specs needs to be a dict.")
            return False
        
        if 'start' in pattern_spec:
            start = pattern_spec['start']
        else:
            start = ['^']
        if type(start) is str:
            start = [start]
        elif type(start) is not list:
            logging.warning(f"Start pattern: {start} has wrong type.")
            return False
        
        if 'end' in pattern_spec:
            end = pattern_spec['end']
        else:
            end = ['$']
        if type(end) is str:
            end = [end]
        elif type(end) is not list:
            logging.warning(f"End pattern: {end} has wrong type.")
            return False
        
        if len(start) != len(end):
            logging.warning(f"Start pattern and end pattern need to have the same length")
            return False
                
        if 'trans_fnct' in pattern_spec:
            trans_fnct = pattern_spec['trans_fnct']
        else:
            identity = lambda x: x
            trans_fnct = [identity for _ in range(len(start))]
        if type(trans_fnct) is not list:
            if callable(trans_fnct):
                trans_fnct = [trans_fnct for _ in range(len(start))]
            else:
                logging.warning(f"Wrong type of trans_fnct")
                return False
        
        if len(start) != len(trans_fnct):
            logging.warning(f"patterns and trans_fnct  need to have the same length")
            return False
        
        if 'must_contain' in pattern_spec:
            must_contain = pattern_spec['must_contain']
        else:
            must_contain = ['' for _ in range(len(start))]
        if type(must_contain) is not list:
            if type(must_contain) is str:
                must_contain = [must_contain for i in range(len(start))]
            else:
                logging.warning(f"Wrong type of must_contain")
                return False
        
        if len(start) != len(must_contain):
            logging.warning(f"patterns and must_contain need to have the same length")
            return False

        
        
        attr_values = []
        for s, e, fnct, mc in zip(start, end, trans_fnct, must_contain):
            if not _contains_patterns(string, mc):
                continue
            attr_value = _extract_string_between_two_patterns(string, s, e)
            if attr_value is not None and attr_value:                    
                attr_value = fnct(attr_value)
                attr_values.append(attr_value)

        if len(attr_values) <= choose_pattern_nr:
            attr_value = None
        else:
            attr_value = attr_values[choose_pattern_nr]
            
        attr_dict[attr] = attr_value
        
        
    return attr_dict


def _print_dicts(dictionary, max_rows=None, max_col_width=120, selection=None, exclude_keys=[]):
    if not dictionary:
        return False

    if selection is not None:
        if type(selection) is list:
            sel = {k: dictionary[k] for k in selection}
        else:
            logging.warning(f"Selection needs to be of type: list.") 
    else:
        sel = dictionary
    
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_colwidth', max_col_width)
    df = pd.DataFrame.from_dict(sel, orient='index')
    df.drop(exclude_keys, axis=1, inplace=True)
    display(df)
    
    return True


def _calc_trimmed_mean(df, columns=[], x_col=None, x_limits=[], percentile=0):
        
    if not columns:
        columns = list(df.columns)

    if not set(columns).issubset(set(df.columns)):
        logging.warning(f"Not all columns found in df!")
        return False

    if x_col is not None and x_limits and len(x_limits) == 2:
        # new_df = df[x_limits[0] <= df[x_col]].copy()
        # new_df = new_df[new_df[x_col] <= x_limits[1]]
        new_df = df[(x_limits[0] <= df[x_col]) & (df[x_col] <= x_limits[1])]
        new_df = new_df.reset_index()
    else:
        new_df = df

    for c in columns:
        trim_limits = new_df[c].quantile([percentile/100, 1-percentile/100])

        trimmed_values = new_df[(trim_limits.iloc[0] <= new_df[c]) & (new_df[c] <= trim_limits.iloc[1])][c]
        trimmed_mean = trimmed_values.mean()

        new_df[c+'_avg'] = pd.Series([trimmed_mean for t in new_df['Time']])

        df = pd.merge(df, new_df[['Time', c+'_avg']], how='outer') 

        print(trimmed_mean)

    return df


def _sanity_check_selection(analyzer):
    if not analyzer.selection:
        logging.warning(f"Define a selection first")
        return False
    
    if analyzer.measurements:
        sel = set(analyzer.selection)
        if not sel.issubset(analyzer.measurements.keys()):
            logging.warning(f"Could not find data for all entries in selection: {analyzer.selection}")
            return False
    else:
        logging.warning(f"No measurements loaded")
        return False

    return True



def _axs_idx_map(idx, num_cols, num_rows, identity=True):
    if identity:
        return idx
    
    axs_idx_col = int(idx % num_cols)
    axs_idx_row = int(np.floor(idx/num_cols))

    if num_rows <= 1:
        return axs_idx_col
    elif num_cols <= 1:
        return axs_idx_row
    else: 
        return (axs_idx_row, axs_idx_col)


def _parse_template(measurement, template=None, exclude_keys=['data'], additional_labels=None):
    format_dict = {}
    default_template = ''
    use_default = False
    if template is None:
        template = ''
        use_default = True
    elif type(template) is not str:
        logging.warning(f'template needs to be a string. Aborting...')
        return False

    if additional_labels is not None:
        if type(additional_labels) is dict:
            measurement_with_labels = measurement.copy()
            measurement_with_labels.update(additional_labels)
            measurement = measurement_with_labels
        else:
            logging.warning(f'additional_labels needs to be a dict. Ignoring...')
            
    for k,v in measurement.items():
        if k in exclude_keys:
            continue
        
        default_template = default_template + f'{k}: {v}, '
        
        r = re.compile(str(k))
        if r.search(template):
            format_dict[k] = v
    
    default_template = default_template[:-2]
    
    if use_default:
        return default_template.format(**format_dict)
    
    try:
        string = template.format(**format_dict)
    except KeyError:
        logging.warning(f'Did not find all attributes specified in the template. Defaulting to default template')
        string = default_template.format(**format_dict)
        
    return string





def _set_tick_number(ax, axis, nticks, scale='linear'):
    if scale == 'linear':
        ax.ticklabel_format(axis=axis, style='sci', scilimits=(-3,3))
        ax.locator_params(axis=axis, nbins=nticks)
        return True
    if scale == 'log':
        if axis == 'x':
            sub_ax = ax.xaxis
        elif axis == 'y':
            sub_ax = ax.yaxis
        else:
            logging.warning(f'axis: {axis} not valid.')
            return False
        locmaj = matplotlib.ticker.LogLocator(base=10, numticks=nticks) 
        sub_ax.set_major_locator(locmaj)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=nticks)
        sub_ax.set_minor_locator(locmin)
        sub_ax.set_minor_formatter(matplotlib.ticker.NullFormatter())

        return True


def _pick_points(dataset, axis, limits, nr_of_rows=None, aggr_fn=None, eps=0.1, head=True):
    if axis not in dataset:
        logging.warning(f"Axis: {axis} not found.")
        return False 

    if type(limits) is int or type(limits) is float:
        sel_data = dataset[dataset[axis].sub(limits).abs() < eps]
    elif type(limits) is list and len(limits) == 2:
        sel_data = dataset[(dataset[axis]>=limits[0]) & (dataset[axis]<=limits[1])]
    else:
        logging.warning(f"limits needs to be of type: int, float or list of length 2. Aborting...")
        return False
    
    if sel_data.empty:
        logging.warning(f"No datapoints found in data for limits: {limits}. Aborting...")
        return False
    
    if aggr_fn is not None:
        if aggr_fn == 'mean':
            sel_data = sel_data.mean(axis=0, numeric_only=True)
        else:
            logging.warning(f"Unknown aggr_fn: {aggr_fn}. Aborting...")
            return False

    elif nr_of_rows is not None:
        if type(nr_of_rows) is not int:
            logging.warning(f"nr_of_rows needs to be an int. Ignoring...")
        else:
            if head:
                sel_data = sel_data.head(nr_of_rows)
            else:
                sel_data = sel_data.tail(nr_of_rows)

    return sel_data


def _add_data_cols(data, value_map:dict, merge_on=None):
    df = pd.DataFrame(value_map)
    if merge_on is None:
        return data.merge(df, left_index=True, right_index=True)
    else:
        return data.merge(df, on=merge_on)


def _generate_safe_path(dir_path, subfolder_name, name):
    # Set save path
    save_dir = dir_path / Path(subfolder_name)
    
    try:
        makedirs(save_dir)
    except FileExistsError:
        pass
    
    name = _clean_string(name)
    
    safepath = save_dir / Path(str(name))
    return safepath

def _clean_string(string):
    string = string.replace(' ', '_')
    string = string.replace(';', '_')
    string = string.replace(',', '_')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace('/', '_')
    string = string.replace(':', '')
    string = string.replace('.', '_')

    return string

def _rename_safepath(dir_path, subfolder_name, name):
    num = 2
    while True:
        new_name = name + f'_{num}'
        new_safepath = _generate_safe_path(dir_path, subfolder_name, new_name)
        if not isfile(str(new_safepath)+'.png'):
            break
        num = num+1
    return new_safepath
    


