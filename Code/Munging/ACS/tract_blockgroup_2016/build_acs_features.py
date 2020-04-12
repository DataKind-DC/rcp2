# Pull out standardized variables from raw ACS files gathered by prep_acs.py

import pandas as pd
import numpy as np
import logging
import argparse
import re
import os
import pdb


def get_var_info(var_list, lu_df):
	""" get variable types and labels from lookup data frame

	:param var_list: list of string variable names
	:param lu_df: column lookup dataframe ready from output of prep_acs.py

	:returns: dict with variable info ... keys=type,label
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

			vars[v] = temp

	return vars


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


def document_variables(data_df, var_dict, transform_df=None):
	""" write out labels and operations for each variable

	:param data_df: pandas dataframe with raw data
	:param var_dict: variable info dict returned from get_var_info
	:param transform_df: pandas dataframe with transformations

	:return: string documentation
	"""

	doc = ''
	for c in list(data_df.columns)[2:]:
		if c in var_dict.keys():
			doc += '{code} = {label}\n'.format(code=c, 
											   label=var_dict[c]['label'])
		else:
			if transform_df is not None:
				if c in transform_df['variable_name'].values:
					row = transform_df.loc[transform_df['variable_name'] == c, \
										   :].to_dict()
					ind = transform_df.index[transform_df['variable_name'] == \
											 c].tolist()[0]
					row = {k: row[k][ind] for k in row}
					raw_arg1 = str(row['argument1'])
					if raw_arg1 in var_dict.keys():
						arg1 = '{label} ({code})'.\
							format(code=raw_arg1, 
								   label=var_dict[raw_arg1]['label'])
					else:
						arg1 = raw_arg1

					raw_arg2 = str(row['argument2'])
					if raw_arg2 in var_dict.keys():
						arg2 = '{label} ({code})'.\
							format(code=raw_arg2, 
								   label=var_dict[raw_arg2]['label'])
					else:
						arg2 = raw_arg2

				doc += '{c} = {a1}'.format(c=c, a1=arg1)
				if row['operator'] == '=':
					doc += '\n'
				else:
					doc += ' {op} {a2}\n'.format(op=row['operator'], a2=arg2)
	return doc


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
		should have column header: variable_name
		if doing transformations, headers should be (tab delimited): 
			variable_name, operator, argument1, argument2
	:param lookup_file: string filename for column lookup file from prep_acs.py
	:param acs_files_path: string path to folder containing raw ACS data
	:param output_file: string name, without extension, for output files

	:returns: None
	"""

	logging.basicConfig(format='%(asctime)s - %(funcName)s - %(message)s',
	                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

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
	if not do_transforms:
		raw_vars = transforms['variable_name'].values.tolist()
	else:
		arg1s = [x for x in transforms['argument1'].astype(str).values.tolist() 
				 if re.match(r'^[BCD][\d_]+$', x)]
		arg2s = [x for x in transforms['argument2'].astype(str).values.tolist() 
				 if re.match(r'^[BCD][\d_]+$', x)]
		raw_vars = list(set(arg1s + arg2s))
	vars = get_var_info(raw_vars, lu)
	all_files, file_to_var = get_file_info(vars, lu)

	# get list of raw files and states
	raw_files = os.listdir(acs_files_path)
	states = list(set([re.sub('^([A-Za-z]+)_.*', '\\1', x) for x in raw_files]))
	states = [x for x in states if x[0].upper() == x[0]]
	states = sorted(states)

	# build df to hold features
	comb = pd.DataFrame()

	# process raw files state by state, doing calculations on each
	for s in states:
		logging.info('processing {state}'.format(state=s))
		st_raw = pd.DataFrame()
		for f in all_files:
			filename = '{fp}/{state}_raw_{file}.csv'.format(fp=acs_files_path,
															state=s, file=f)
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
		if do_transforms:
			st_raw = do_transformations(st_raw, transforms)

		comb = comb.append(st_raw)
	comb = comb.reset_index()

	comb['geoid'] = ('#_' + comb['geoid']).where(comb['geoid'].str[:2] != '#_')

	comb.drop('index', axis='columns', inplace=True)
	comb.to_csv(output_file + '.csv', index=False)
	logging.info('features written to {0}.csv'.format(output_file))

	# write out variables
	info = document_variables(comb, vars, transform_df=transforms)
	var_out_file = output_file + '_variables.csv'
	with open(var_out_file, 'w') as o:
		o.write(info)
	logging.info('variables written to {0}'.format(var_out_file))


def build_acs_features_cmd():

    parser = argparse.ArgumentParser(description='Build ACS features')

    parser.add_argument('vars_file', type=str, help='file with list of vars' 
    					' or list of transformations')
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
