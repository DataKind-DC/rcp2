import pandas as pd
import numpy as np
import os
import requests
import logging
import argparse
import re
import pathlib

API_KEY = '62259668f45cf937bb592cd9236be97bc6bc5031'

MAX_VARS = 50

STATE_CODES = {'Alabama': ('AL', '01'), 'Alaska': ('AK', '02'),
               'Arizona': ('AZ', '04'), 'Arkansas': ('AR', '05'),
               'California': ('CA', '06'), 'Colorado': ('CO', '08'),
               'Connecticut': ('CT', '09'), 'Delaware': ('DE', '10'),
               'District of Columbia': ('DC', '11'), 'Florida': ('FL', '12'),
               'Georgia': ('GA', '13'), 'Hawaii': ('HI', '15'), 
               'Idaho': ('ID', '16'), 'Illinois': ('IL', '17'), 
               'Indiana': ('IN', '18'), 'Iowa': ('IA', '19'), 
               'Kansas': ('KS', '20'), 'Kentucky': ('KY', '21'), 
               'Louisiana': ('LA', '22'), 'Maine': ('ME', '23'), 
               'Maryland': ('MD', '24'), 'Massachusetts': ('MA', '25'),
               'Michigan': ('MI', '26'), 'Minnesota': ('MN', '27'), 
               'Mississippi': ('MS', '28'), 'Missouri': ('MO', '29'), 
               'Montana': ('MT', '30'), 'Nebraska': ('NE', '31'), 
               'Nevada': ('NV', '32'), 'New Hampshire': ('NH', '33'), 
               'New Jersey': ('NJ', '34'), 'New Mexico': ('NM', '35'), 
               'New York': ('NY', '36'), 'North Carolina': ('NC', '37'), 
               'North Dakota': ('ND', '38'), 'Ohio': ('OH', '39'), 
               'Oklahoma': ('OK', '40'), 'Oregon': ('OR', '41'), 
               'Pennsylvania': ('PA', '42'), 'Rhode Island': ('RI', '44'), 
               'South Carolina': ('SC', '45'), 'South Dakota': ('SD', '46'), 
               'Tennessee': ('TN', '47'), 'Texas': ('TX', '48'), 
               'Utah': ('UT', '49'), 'Vermont': ('VT', '50'), 
               'Virginia': ('VA', '51'), 'Washington': ('WA', '53'), 
               'West Virginia': ('WV', '54'), 'Wisconsin': ('WI', '55'),
               'Wyoming': ('WY', '56')}


def extract_vars(vars_file):
    """ extract vars to be downloaded from munging file
    
    :param vars_file: string path and filename to list of variables
        should have column header: variable_name
        if doing transformations, headers should be (tab delimited): 
            variable_name, operator, argument1, argument2
    
    :returns: list of sorted variable names
    """
    # get vars/transform file
    do_transforms = False
    try:
        transforms = pd.read_csv(vars_file, sep='\t')
    except:                                                                   
        logging.error('unable to read {0}'.format(vars_file))
        return
    if 'variable_name' not in transforms.columns:
        logging.error('missing variable_name column')
        return
    if len(transforms.columns) > 1:
        req_cols = ['operator', 'argument1', 'argument2']
        if len([x for x in req_cols if x in transforms.columns]) != \
            len(req_cols):
            logging.warn('missing some required columns for transforms.'
                         ' will pull raw variables instead')
        else:
            do_transforms = True

    # get vars into a list
    try:
        f = open(vars_file, 'r')
        acs_vars = f.readlines()
        acs_vars = [vars.strip() for vars in acs_vars if vars.strip()]
    except:
        logging.error('unable to read vars file {0}'.format(vars_file))
        return

    # get variable and file info
    if not do_transforms:
        raw_vars = transforms['variable_name'].values.tolist()
    else:
        arg1s = [x for x in transforms['argument1'].astype(str).values.tolist() 
                 if re.match(r'^[BCD][\d_]+$', x)]
        arg2s = [x for x in transforms['argument2'].astype(str).values.tolist() 
                 if re.match(r'^[BCD][\d_]+$', x)]
        raw_vars = list(set(arg1s + arg2s))
    raw_vars = sorted(raw_vars)
    return transforms, raw_vars
    
    
def do_transformations(data_df, transform_df):
    """ perform transformation on data frame as listed in transform data frame

    The transform dataframe should have four columns:
        new_variable, operator, argument1, argument 2

            new_variable is the variable to create
            operator is the operation to perform
                currently supports: -, +, *, /, =
            argument1/2 are either column names or scalar values.

    Operations are performed sequentially so you can include a new variable
    created via the transforms as an argument in a later calculation.

    :param data_df: pandas dataframe with raw data
    :param transform_df: pandas dataframe with transformations

    :return: transformed pandas dataframe	
    """

    ok_operators = {'-', '+', '*', '/', '='}
    transform_df['operator'] = transform_df['operator'].str.strip()

    for r in transform_df.itertuples():
        if r.operator in ok_operators:
            try:
                rarg1 = str(r.argument1).strip()
                rarg2 = str(r.argument2).strip()
                if rarg1 in data_df.columns:
                    arg1 = data_df[rarg1]
                else:
                    arg1 = float(rarg1)
                if rarg2 in data_df.columns:
                    arg2 = data_df[rarg2]
                elif rarg2.lower() != 'nan':
                    arg2 = float(rarg2)
                else:
                    arg2 = ''

                if r.operator == '-':
                    data_df[r.variable_name] = arg1 - arg2
                elif r.operator == '+':
                    data_df[r.variable_name] = arg1 + arg2
                elif r.operator == '*':
                    data_df[r.variable_name] = arg1 * arg2
                elif r.operator == '=':
                    data_df[r.variable_name] = arg1
                elif r.operator == '/':
                    if r.argument2 and r.argument2 != 'NaN':
                        if isinstance(arg2, float):
                            data_df[r.variable_name] = arg1 / arg2
                        else:
                            valid_mask = (data_df[r.argument2].notnull()) & \
                                         (data_df[r.argument2] != 0)
                            data_df[r.variable_name] = (arg1 / arg2).\
                                where(valid_mask)
                            data_df.loc[~valid_mask, r.variable_name] = np.nan

                    else:
                        logging.warning('tried to do bad division of {0}/{1}'.\
                            format(r.argument1, r.argument2))
            except:
                logging.warning('invalid argument(s): {0}, {1}'.\
                                format(r.argument1, r.argument2))

    return data_df

def get_vars_lookup(year, vars, transforms):
    """ downloads variable template data from census
    
    :param year: string four-digit year
    :param vars: list of strings of var names
    :param transforms: DataFrame of transformations 

    :return: list of vars and their transformations
    """

    url = ("https://api.census.gov/data/{y}/acs/acs5/variables").format(y=year)
    r = requests.get(url)
    response = r.json()

    # remove header
    all_vars = response[4:]
 
    # sort by var
    all_vars.sort(key=lambda g: g[0])

    vars_dict = dict()
    for name, label, concept in all_vars:
        label = label.replace("!!", " ")
        name = name[:-1]
        vars_dict[name] = '{c}: {l}'.format(c=concept,l=label)

    vars_labels = [var + ' = ' + vars_dict[var] for var in vars]

    transforms = transforms.where(pd.notnull(transforms), None)
    transformations = list()
    for index, row in transforms.iterrows():
        vn = row['variable_name']
        op = row['operator']
        arg1 = row['argument1']
        arg2 = row['argument2']
        if arg2==None:
            t = "{vn} = {vd} ({a1})".format(vn=vn, vd=vars_dict[arg1], a1=arg1)
        else:
            if arg1[0] == 'B':
                arg1 = vars_dict[arg1] + "({0})".format(arg1)
            if arg2[0] == 'B':
                arg2 = vars_dict[arg2] + "({0})".format(arg2)
            t = "{vn} = {a1} {op} {a2}".format(vn=vn, a1=arg1, op=op, a2=arg2)
        transformations.append(t)
    vars_transforms = vars_labels + transformations
    return vars_transforms

def get_vars_data(vars, state_code, year):
    """ downloads variable data from census
    
    :param vars: list of strings of var names
    :param state_code: string two digit state code
    :param year: string four-digit year

    :return: pandas dataframe of geoid and vars data
    """

    logging.info('downloading vars {0} to {1}'.format(vars[0], vars[-1]))

    # add E at the end for estimate vars
    cols = [var + 'E' for var in vars]
    
    # convert list of vars to string for api
    cols = ','.join(cols)


    # wait 1 min, 2 min and 5 min for each api call
    timeouts = [60,120,300]
    for wait in timeouts:
        try:
            url = ('https://api.census.gov/data/{y}/acs/acs5?get=' +
                '{v}' + '&for=block%20group:*&in=state:{sc}%20county:*&key={k}').\
                format(v=cols, sc=state_code, y=year, k=API_KEY)
            r = requests.get(url, timeout=wait)    
            vars_data = r.json()
            break
        except:
            continue
    
    if vars_data is None:
        logging.error('unable to download vars')
        return
    
    # remove header 
    vars_data = vars_data[1:]

    # create geoids and sort by them
    vars_data = [[''.join(data[-4:]), *data[:-4]] for data in vars_data]
    vars_data.sort(key=lambda g: g[0])

    vars_data = pd.DataFrame(vars_data, columns=['geoid',*vars])
    
    for var in vars:
        vars_data[var] = vars_data[var].astype('float64')

    return vars_data

def combine_features(state_names, year, output_path):
    """ combine individual csv state files into one

    :param state_names: list of state names
    :param year: string four-digit year
    :param output_path: string path to write raw files to
    
    :return: None
    """

    logging.info('Combining all states into one file')
    if state_names[0] == 'all':
        state_names = sorted(STATE_CODES)
    
    output_path = output_path.format(year=year)
    output_file = output_path + 'acs_{y}_data.csv'.format(y=year)
    
    if os.path.isfile(output_file):
        os.remove(output_file)
    
    df = pd.DataFrame()
    for state in sorted(state_names):
        logging.info('Adding {state}'.format(state=state))

        temp = pd.read_csv(output_path + state + '.csv')
        with open(output_file, mode = 'a') as f:
            temp.to_csv(f, header=f.tell()==0,index = False)
        os.remove(output_path + state + '.csv')
        
    return

def acs_full(year, vars_file, output_path):
    """ download all ACS data needed for model for a given year 

    :param year: string four-digit year
    :param vars_file: string path and filename to list of variables
        should have column header: variable_name
        if doing transformations, headers should be (tab delimited): 
        variable_name, operator, argument1, argument2

    :return: None
    """

    state_names = sorted(STATE_CODES)
    output_path = output_path.format(year=year)

    # check to see which states have already been downloaded    

    if not pathlib.Path(output_path).exists():
        os.mkdir(output_path)

    completed_states = os.listdir(path=output_path)
    completed_states = [state[:-4] for state in completed_states]
    states = sorted(list(set(state_names) - set(completed_states)))


    for state in states:
        acs_main(state, year, vars_file, output_path)
    return


def acs_main(state, year, vars_file, output_path):
    """ download ACS data needed for model and do transformations 

    :param state: string state name
    :param year: string four-digit year
    :param vars_file: string path and filename to list of variables
        should have column header: variable_name
        if doing transformations, headers should be (tab delimited): 
        variable_name, operator, argument1, argument2
    :param output_path: string path to write raw files to
    
    :returns: None
    """

    logging.info('downloading {st}'.format(st=state))

    state_data = pd.DataFrame()
    state_code = STATE_CODES[state][1]

    output_path = output_path.format(year=year)
    file_name = state + '.csv'
    output_file = output_path + file_name
    column_lookup = output_path + "acs_{y}_features_variables.csv".format(y=year)

    if not pathlib.Path(output_path).exists():
        os.mkdir(output_path)

    # variables to be downloaded for transformations
    transforms,vars = extract_vars(vars_file)

    # create vars and transformation lookup file
    if not pathlib.Path(column_lookup).exists():
        logging.info('creating {cl}'.format(cl=column_lookup))
        vars_lookup = get_vars_lookup(year, vars.copy(),transforms)
        with open(column_lookup, 'w') as f:
            f.write('\n'.join(vars_lookup))

    
    # call api in batches of 50 vars (api limit) at a time 
    batch_vars = [vars[i:i+MAX_VARS] for i in range(0, len(vars), MAX_VARS)]

    for vars in batch_vars:
        # 50 cols of data for all rows
        vars_data = get_vars_data(vars, state_code, year)
        if vars_data is None:
            return
        if state_data.empty:
            state_data = vars_data
        else:
            state_data = pd.merge(state_data, vars_data, on='geoid')
    
    # make lookup easier
    state_data['geoid'] = state_data['geoid'].apply(lambda g : '#_' + g)
    state_data.insert(0, 'state', STATE_CODES[state][0])
    
    # transforms
    transforms = pd.read_csv(vars_file, sep='\t')
    state_data = do_transformations(state_data, transforms)
    state_data.to_csv(output_file, index=False)
    
    logging.info('{st} data downloaded'.format(st=state))

    return

def acs_cmd():
    parser = argparse.ArgumentParser(description='Prep ACS data')
    parser.add_argument('year', type=str, help='four digit year')
    parser.add_argument('vars_file', type=str, help='file with list of vars'
                    ' or list of transformations')
    parser.add_argument('state', type=str, help='state full name', nargs='+')

    parser.add_argument('-a', '--all', default=False, action='store_true',
        help='download data for all states for a given year')
    parser.add_argument('-op', '--output_path', default='../../../Data/ACS_Extraction/',
        help='path to write files to')
    parser.add_argument('-cf', '--combine_features', default=False,
    action='store_true', help='create file of combined state data file then delete state files')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(funcName)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    num_logger = logging.getLogger('numexpr')
    num_logger.setLevel(logging.ERROR)

    if args.all:
        acs_full(args.year, args.vars_file, output_path=args.output_path)
    else:
        for state in args.state:
            acs_main(state, args.year, args.vars_file, args.output_path)
    
    if args.combine_features:

        combine_features(args.state, args.year, args.output_path)

if __name__ == '__main__':
    acs_cmd()