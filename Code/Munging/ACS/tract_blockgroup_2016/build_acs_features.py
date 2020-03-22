# Pull out standardized variables from raw ACS files gathered by prep_acs.py

import pandas as pd
import numpy as np
import logging
import argparse
import re
import os
import pdb


def get_var_info(var_list, lu_df):
	""" determine appropriate denominators for creating features

	:param var_list: list of string variable names
	:param lu_df: column lookup dataframe ready from output of prep_acs.py

	:returns: dict with variable info ... keys=type,label,denom,denom_label
	"""

	vars = {v: {} for v in var_list}

	for v in vars.keys():
		if v not in lu_df['code'].values:
			logging.warning('{0} not a valid code'.format(v))
			del vars[v]
		else:
			# get var type and denominator
			temp = {'denom': '', 'denom_label': ''}
			temp['type'] = lu_df.loc[lu_df['code'] == v, 'type'].values[0]
			temp['label'] = lu_df.loc[lu_df['code'] == v, 'label'].values[0]

			if temp['type'] == 'Int32':
				if v[-3:] != '001':
					temp_denom = v[:-3] + '001'
					if temp_denom in lu_df['code'].values:
						temp['denom'] = temp_denom
						temp['denom_label'] = lu_df.loc[lu_df['code'] == \
														temp_denom, 'label'].\
														values[0]
					elif v != 'B00001_001':
						temp['denom'] = 'B00001_001'
						temp['denom_label'] = lu_df.loc[lu_df['code'] == \
														temp_denom, 'label'].\
														values[0]
				else:
					temp['denom'] = 'B00001_001'
					temp['denom_label'] = lu_df.loc[lu_df['code'] == \
													temp_denom, 'label'].\
													values[0]
			vars[v] = temp

	return vars


def get_file_info(var_dict, lu_df):
	""" determine which raw files are needed to pull specified variales

	:param var_dict: variable info dict returned from get_var_info
	:param lu_df: column lookup dataframe ready from output of prep_acs.py

	:returns: list of necessary files, dict of vars per file
	"""

	all_vars = list(set(list(var_dict.keys()) + \
					    [var_dict[x]['denom'] for x in var_dict.keys()]))
	all_vars = [x for x in all_vars if x]

	file_to_var = {}
	for v in all_vars:
		file_num = lu_df.loc[lu_df['code'] == v, 'output_part_num'].values[0]
		if file_num in file_to_var.keys():
			temp = file_to_var[file_num]
			file_to_var[file_num] = temp + [v]
		else:
			file_to_var[file_num] = [v]
	all_files = list(file_to_var.keys())

	return all_files, file_to_var


def build_acs_features_main(vars_file, lookup_file, acs_files_path, 
						  output_file='acs_features'):
	""" create standardized features from raw ACS data

	:param vars_file: string path and filename to list of variables
	:param lookup_file: string filename for column lookup file from prep_acs.py
	:param acs_files_path: string path to folder containing raw ACS data

	:returns: None
	"""

	logging.basicConfig(format='%(asctime)s - %(funcName)s - %(message)s',
	                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

	# get vars into a list
	try:
		f = open(vars_file, 'r')
		acs_vars = f.readlines()
		acs_vars = [x.strip() for x in acs_vars if x.strip()]
	except:
		logging.error('unable to read vars file {0}'.format(vars_file))
		return

	# read column lookup file
	try:
		lu = pd.read_csv(lookup_file)
	except:                                                                   
		logging.error('unable to read {0}'.format(lookup_file))
		return

	# get variable and file info
	vars = get_var_info(acs_vars, lu)
	all_files, file_to_var = get_file_info(vars, lu)

	# get list of raw files and states
	raw_files = os.listdir(acs_files_path)
	states = list(set([re.sub('^([A-Za-z]+)_.*', '\\1', x) for x in raw_files]))
	states = [x for x in states if x[0].upper() == x[0]]
	states = sorted(states)

	# build df to hold features
	cols = ['state', 'geoid'] + sorted(list(vars.keys()))
	comb = pd.DataFrame(columns=cols)

	# process raw files state by state, doing calculations on each
	for s in states:
		logging.info('processing {state}'.format(state=s))
		st_raw = pd.DataFrame()
		for f in all_files:
			filename = 'acs_output/{state}_raw_{file}.csv'.format(state=s, 
																  file=f)
			rel_cols = ['state', 'geoid'] + file_to_var[f]
			col_types = {x: float for x in file_to_var[f]}
			col_types.update({'state': str, 'geoid': str})
			temp = pd.read_csv(filename, usecols=rel_cols, dtype=col_types)
			if st_raw.shape[0] == 0:
				st_raw = temp.copy()
			else:
				st_raw = pd.merge(st_raw, temp, how='outer', 
								  on=['state', 'geoid'])

		# do the calculations on the current state
		st_raw['state'] = st_raw['state'].str.upper()
		for v in vars.keys():
			denom = vars[v]['denom']
			if denom:
				valid_mask = (st_raw[denom].notnull()) & (st_raw[denom] != 0)
				st_raw[v] = (st_raw[v] / st_raw[denom]).where(valid_mask)
				st_raw.loc[~valid_mask, v] = np.nan
		st_raw = st_raw[cols]

		comb = comb.append(st_raw)
	comb = comb.reset_index()
	comb.drop('index', axis='columns', inplace=True)
	comb.to_csv(output_file + '.csv', index=False)
	logging.info('features written to {0}.csv'.format(output_file))

	# write out transformations
	with open(output_file + '_transforms.csv', 'w') as o:
		for v in vars.keys():
			line = '{code} = {label}'.format(code=v, label=vars[v]['label'])
			if vars[v]['denom']:
				line += ' ... DIVIDED BY ... {dlabel} ({code})'.\
					format(dlabel=vars[v]['denom_label'], 
						   code=vars[v]['denom'])
			o.write(line + '\n')
	logging.info('transformations written to {0}_transfors.csv'.\
		format(output_file))


def build_acs_features_cmd():

    parser = argparse.ArgumentParser(description='Build ACS features')

    parser.add_argument('vars_file', type=str, help='file with list of vars')
    parser.add_argument('lookup_file', type=str, help='column lookup filename')
    parser.add_argument('acs_files_path', type=str, 
    					help='folders with raw data')

    parser.add_argument('-o', '--output_file', default='acs_features',
                        help='output file name (default %(default)s)')

    args = parser.parse_args()
    build_acs_features_main(args.vars_file, args.lookup_file, 
    						args.acs_files_path, output_file=args.output_file)


if __name__ == '__main__':
    build_acs_features_cmd()
