### Download and prep raw ACS data

import pandas as pd
import os
import requests
import logging
import zipfile
import argparse
import re
import pathlib

def build_column_lookup(template_folder=None):
    """ create lookup from code to variable label from template data

    :param template_folder: string folder path
        default: None (will download data from Census webiste)

    :return: pandas dataframe of columns, template folder
    """

    if not template_folder:
        try:
            url = 'https://www2.census.gov/programs-surveys/acs/' + \
                'summary_file/2016/data/2016_5yr_Summary_FileTemplates.zip'
            r = requests.get(url)
            open('templates.zip', 'wb').write(r.content)
            with zipfile.ZipFile('templates.zip', 'r') as zip_ref:
                zip_ref.extractall()

            template_folder = './templates/'
            logging.info('templates downloaded')
        except:
            logging.error('unable to get template file from web')
            return None, None

    col_lookup = pd.DataFrame(columns=['code', 'label', 'file_num', 'field_num', 
        'type', 'output_part_num'])
    for i in range(1,123):
        temp = pd.read_excel('{0}/seq{1}.xls'.format(template_folder, i))
        data_cols = list(temp.columns)[6:]
        temp = temp[data_cols]
        temp = pd.melt(temp, value_vars=data_cols, var_name='code', 
            value_name='label')
        temp['file_num'] = i
        temp['field_num'] = temp.index + 1
        temp['type'] = 'Int32'

        col_lookup = col_lookup.append(temp)

    col_lookup = col_lookup[['file_num', 'field_num', 'output_part_num', 'type', 
        'code', 'label']].copy()

    return col_lookup, template_folder


def get_state_data(state):
    """ download state data from Census web site

    :param state: string state name (full name)

    :return: string state path
    """

    try:
        url = ('https://www2.census.gov/programs-surveys/acs/' + 
            'summary_file/2016/data/5_year_by_state/' + 
            '{0}_Tracts_Block_Groups_Only.zip').format(state)
        r = requests.get(url)
        folder_name = '{0}.zip'.format(state)
        open(folder_name, 'wb').write(r.content)
        with zipfile.ZipFile(folder_name, 'r') as zip_ref:
            zip_ref.extractall('{0}'.format(state))

        state_path = './{0}/'.format(state)
        logging.info('{0} files downloaded from web'.format(state))
        return state_path
    except:
        logging.error('unable to get {0} ACS files from web'.format(state))
        return


def build_geo_lookup(state_path, template_folder):
    """ create lookup from logrecno to geoid

    :param state_path: string path to state folder
    :param template_folder: string folder path

    :return: pandas dataframe
    """

    try:
        geo_header = pd.read_excel('{0}/2016_SFGeoFileTemplate.xls'.format(
            template_folder))
        state_files = os.listdir(state_path)
        geo_file = [f for f in state_files if re.match('g20165..\.csv', f)][0]
        geo_lookup = pd.read_csv(state_path + geo_file, 
            names=geo_header.columns, encoding='latin-1', low_memory=False)
        geo_lookup = geo_lookup.loc[~geo_lookup['TRACT'].isnull(), :].copy()

        geo_lookup['STATE'] = geo_lookup['STATE'].astype(int).astype(
            str).str.zfill(2)
        geo_lookup['COUNTY'] = geo_lookup['COUNTY'].astype(int).astype(
            str).str.zfill(3)
        geo_lookup['TRACT'] = geo_lookup['TRACT'].astype(int).astype(
            str).str.zfill(6)

        geo_lookup['geoid'] = geo_lookup['STATE'] + geo_lookup['COUNTY'] + \
            geo_lookup['TRACT']

        geo_block = geo_lookup.loc[~geo_lookup['BLKGRP'].isnull(), :].copy()
        geo_block['geoid'] = geo_block['geoid'] + \
            geo_block['BLKGRP'].astype(int).astype(str)
        geo_block = geo_block[['LOGRECNO', 'geoid']].copy()

        geo_lookup = geo_lookup.loc[geo_lookup['BLKGRP'].isnull(), 
            ['LOGRECNO', 'geoid']].copy()
        geo_lookup = geo_lookup.append(geo_block)

        geo_lookup = geo_lookup.rename({'LOGRECNO': 'logrecno'}, axis='columns')
        geo_lookup['logrecno'] = geo_lookup['logrecno'].astype(int)

        return geo_lookup

    except Exception as ex:
        logging.error('unable to process geo file in {0}'.format(state_path))
        logging.error(str(ex))



def build_raw_file(state, col_lookup, geo_lookup, state_path=None, 
        check_types=False, max_vars=2000, output_path='acs_output/'):
    """ download and combine acs files for a state into one raw csv file

    :param state: string state name (full name)
    :param col_lookup: pandas dataframe created by build_col_lookup
    :param geo_lookup: pandas dataframe created by build_geo_lookup
    :param state_path: string path to state folder
        default: None ... will download data from Census website
    :param check_types: True/False check data types in raw data and update 
        col_lookup
        should be run on first state and then can be skipped
    :param max_vars: number of variables to output per file
        default: 2000
        there are over 22,000 variables total which takes hours to export as one
            big file
    :param output_path: string path to write raw files to
        default: acs_output/

    :return: updated col_lookup dataframe
    """

    logging.info('building raw files for {0}'.format(state))

    std_cols = ['ignore', 'acs_type', 'state', 'ignore2', 'file_num', 
        'logrecno']
    std_types = {'ignore':'str', 'acs_type':'str', 'state':'str', 
        'ignore2':'int', 'file_num':'int', 'logrecno':'int'}

    if not state_path:
        state_path = get_state_data(state)

    files = os.listdir(state_path)
    files = [f for f in files if len(f) >= 19 and f[0]=='e']
    files = sorted(files)

    data = pd.DataFrame()

    num_vars = 0
    part_num = 1
    for file_num, file_name in enumerate(files):
        file_num += 1
        logging.debug('...file {0}'.format(file_num))
        
        # set up cols
        cur_dict = col_lookup.loc[col_lookup['file_num'] == file_num, 
            ['code', 'type', 'label']]
        data_cols = cur_dict['code'].to_list()
        labels = dict(zip(cur_dict['code'], cur_dict['label']))

        # read data and update types if needed
        if check_types:
            temp = pd.read_csv(state_path + '/' + file_name, 
                names=std_cols + data_cols, low_memory=False, 
                index_col=False, na_values='.', encoding='latin-1')
            for c in data_cols:
                if re.match('^MEDIAN .*', labels[c]) or \
                    re.match('^AGGREGATE .*', labels[c]):
                    col_lookup.loc[col_lookup['code'] == c, 'type'] = 'float64'
                else:
                    try:
                        temp[c] = temp[c].astype('Int32')
                    except:
                        col_lookup.loc[col_lookup['code'] == c, 
                            'type'] = 'float64'
        else:
            dtypes = dict(zip(cur_dict['code'], cur_dict['type'])) 
            dtypes.update(std_types)

            try:
                temp = pd.read_csv(state_path + '/' + file_name, 
                    names=std_cols + data_cols, low_memory=False, 
                    index_col=False, na_values='.', dtype=dtypes, 
                    encoding='latin-1')
            except:
                logging.error('unable to read {0}/{1}'.format(
                    state_path, file_name))
                break

        # clean up data
        temp = temp[['state', 'logrecno'] + data_cols]
        temp['logrecno'] = temp['logrecno'].astype(int)
        temp['state'] = temp['state'].astype(str)
        
        if check_types:
            col_lookup.loc[col_lookup['file_num'] == file_num, 
                'output_part_num'] = part_num

        # combine
        if data.shape[0] > 0:
            if len(data_cols) + num_vars <= (1.1 * max_vars):
                data = pd.merge(data, temp, how='outer', 
                    on=['state', 'logrecno'])
                num_vars += len(data_cols)
            else:
                data.to_csv('{output}{0}_raw_{1}.csv'.format(state, part_num, 
                    output=output_path), index=False)
                logging.info('{cols} columns output to {0}_raw_{1}.csv'.format(
                    state, part_num, cols=num_vars))
                num_vars = 0
                part_num += 1
                data = pd.DataFrame()
                data = temp.copy()
                data = pd.merge(data, geo_lookup, how='left', on='logrecno')
                data = data[['state', 'geoid', 'logrecno'] + data_cols]
                if check_types:
                    col_lookup.loc[col_lookup['file_num'] == file_num, 
                        'output_part_num'] = part_num
        else:
            data = temp.copy()
            data = pd.merge(data, geo_lookup, how='left', on='logrecno')
            data = data[['state', 'geoid', 'logrecno'] + data_cols]
        del temp

    if num_vars > 0:
        data.to_csv('{output}{0}_raw_{1}.csv'.format(state, part_num, 
            output=output_path), index=False)
        logging.info('{cols} columns output to {0}_raw_{1}.csv'.format(
            state, part_num, cols=num_vars))

    if check_types:
        col_lookup.to_csv('col_lookup.csv', index=False)
        logging.info('updated column lookup written to col_lookup.csv')


def prep_acs_main(state, template_folder=None, state_path=None, 
        check_types=False, max_vars=2000, output_path='acs_output/'):
    """ prep ACS data

    :param state: string state name (full name)
    :param template_folder: string folder path
        default: None (will download data from Census webiste)
    :param state_path: string path to state folder
        default: None ... will download data from Census website
    :param check_types: True/False check data types in raw data and update 
        col_lookup
        should be run on first state and then can be skipped
    :param max_vars: number of variables to output per file
        default: 2000
        there are over 22,000 variables total which takes hours to export as one
            big file
    :param output_path: string path to write raw files to
        default: acs_output/

    :return: None
    """

    logging.basicConfig(format='%(asctime)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    logging.info('start of prep_acs for {0}'.format(state))

    # make sure output path exists
    pathlib.Path('./' + output_path).mkdir(parents=True, exist_ok=True)

    # get column lookup
    if check_types:
        col_lu, template_folder = build_column_lookup(template_folder)
        if col_lu is None:
            return
    else:
        col_lu = pd.read_csv('col_lookup.csv')

    # get state data
    if not state_path:
        state_path = get_state_data(state)
        if not state_path:
            return

    # set up geoids
    logging.info('set up geoids')
    geo_lu = build_geo_lookup(state_path, template_folder)
    if geo_lu is None:
        return

    # process raw data
    col_lu = build_raw_file(state, col_lu, geo_lu, state_path=state_path, 
        check_types=check_types, max_vars=max_vars, output_path=output_path)

    logging.info('prep_acs completed for {0}'.format(state))


def prep_acs_cmd():

    parser = argparse.ArgumentParser(description='Prep ACS data')
    parser.add_argument('state', type=str, help='state full name', nargs='+')

    parser.add_argument('-tf', '--template_folder', default=None,
        help='template folder path. if None, will download to templates/')
    parser.add_argument('-sp', '--state_path', default=None,
        help='raw state data path. if None, will download to state folder')
    parser.add_argument('-ct', '--check_types', default=False, 
        action='store_true', help='check data types on load')
    parser.add_argument('-mv', '--max_vars', default=2000, type=int,
        help='approximate number of variables to output per file. '  
        'the script allows up to 10 pct more to keep the number of files down')
    parser.add_argument('-op', '--output_path', default='acs_output/',
        help='path to write raw files to')

    args = parser.parse_args()
    for st in args.state:
        prep_acs_main(st, template_folder=args.template_folder, 
            state_path=args.state_path, check_types=args.check_types,
            max_vars=args.max_vars, output_path=args.output_path)


if __name__ == '__main__':
    prep_acs_cmd()

