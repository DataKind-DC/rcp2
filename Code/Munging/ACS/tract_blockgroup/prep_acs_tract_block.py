### Download and prep raw ACS data

import pandas as pd
import os
import requests
import logging
import zipfile
import argparse
import re
import pathlib
import shutil

STATE_NAMES = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
               'Colorado', 'Connecticut', 'Delaware', 'DistrictOfColumbia', 
               'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 
               'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 
               'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
               'Missouri', 'Montana', 'Nebraska', 'Nevada', 'NewHampshire', 
               'NewJersey', 'NewMexico', 'NewYork', 'NorthCarolina', 
               'NorthDakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 
               'RhodeIsland', 'SouthCarolina', 'SouthDakota', 'Tennessee', 
               'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 
               'WestVirginia', 'Wisconsin', 'Wyoming']

def build_column_lookup(year, template_folder=None):
    """ create lookup from code to variable label from template data

    :param year: int four-digit year
    :param template_folder: string folder path
        default: None (will download data from Census webiste)

    :return: pandas dataframe of columns, template folder
    """

    if not template_folder:
        try:
            url = ('https://www2.census.gov/programs-surveys/acs/' + 
                'summary_file/{y}/data/{y}_5yr_Summary_FileTemplates.zip').\
                format(y=year)
            if year == 2010:
                url = url.replace('Summary_FileTemplates', 
                                  'SummaryFileTemplates')
            r = requests.get(url)
            open('templates.zip', 'wb').write(r.content)
            with zipfile.ZipFile('templates.zip', 'r') as zip_ref:
                zip_ref.extractall('templates_{y}'.format(y=year))

            template_folder = './templates_{y}'.format(y=year)
            logging.info('templates downloaded')
        except:
            logging.error('unable to get template file from web')
            return None, None

    col_lookup = pd.DataFrame(columns=['code', 'label', 'file_num', 'field_num', 
        'type', 'output_part_num'])

    if os.path.isdir('{0}/{1}'.format(template_folder, 'seq')):
        temp_folder = '{0}/{1}'.format(template_folder, 'seq')
    elif os.path.isdir('{0}/{1}'.format(template_folder, 'templates')):
        temp_folder = '{0}/{1}'.format(template_folder, 'templates')
        template_folder = temp_folder
    else:
        temp_folder = template_folder

    template_files = os.listdir(temp_folder)
    template_files = [f for f in template_files if \
                      re.match('seq\d+\.xlsx*', f, re.IGNORECASE)]
    for f in template_files:
        seq_num = int(re.sub('seq(\d+)\.xlsx*', '\\1', f, flags=re.IGNORECASE))
        temp = pd.read_excel('{tf}/{f}'.format(tf=temp_folder, f=f))
        data_cols = list(temp.columns)[6:]
        temp = temp[data_cols]
        temp = pd.melt(temp, value_vars=data_cols, var_name='code', 
            value_name='label')
        temp['file_num'] = seq_num
        temp['field_num'] = temp.index + 1
        temp['type'] = 'Int32'

        col_lookup = col_lookup.append(temp)

    col_lookup = col_lookup[['file_num', 'field_num', 'output_part_num', 'type', 
        'code', 'label']].copy()

    os.remove('templates.zip')

    return col_lookup, template_folder


def get_state_data(state, year):
    """ download state data from Census web site

    :param state: string state name (full name)
    :param year: int four-digit year

    :return: string state path
    """

    try:
        url = ('https://www2.census.gov/programs-surveys/acs/' + 
            'summary_file/{y}/data/5_year_by_state/' + 
            '{st}_Tracts_Block_Groups_Only.zip').format(y=year, st=state)
        r = requests.get(url)
        folder_name = '{0}.zip'.format(state)
        open(folder_name, 'wb').write(r.content)
        with zipfile.ZipFile(folder_name, 'r') as zip_ref:
            zip_ref.extractall('{0}'.format(state))

        state_path = './{0}/'.format(state)
        logging.info('{0} files downloaded from web'.format(state))
        return state_path
    except:
        logging.error('unable to get {st} {y} ACS files from web'.\
                      format(st=state, y=year))
        return


def build_geo_lookup(state_path, template_folder, year):
    """ create lookup from logrecno to geoid

    :param state_path: string path to state folder
    :param template_folder: string folder path
    :param year: int four-digit year

    :return: pandas dataframe
    """

    # get header file, accounting for multiple folder structures, xls vs xlsx
    header_file_stub = '{y}_SFGeoFileTemplate.xls'.format(y=year)
    check_locations = ['{tf}/{f}', '{tf}/{f}x', '{tf}/templates/{f}', 
                       '{tf}/templates/{f}x']
    header_path = None
    for loc in check_locations:
        loc_str = loc.format(tf=template_folder, f=header_file_stub)
        if os.path.exists(loc_str):
            header_path = loc_str
            break
    if not header_path:
        logging.error('unable to find {} in {}'\
                      .format(header_file_stub, template_folder))
        return

    # read and process geographic lookup file
    try:
        geo_header = pd.read_excel(header_path)
        state_files = os.listdir(state_path)
        geo_file = sorted([f for f in state_files if \
                    re.match('g{y}5..\....'.format(y=year), f)])[0]
        if geo_file[-3:] == 'txt':
            sep = '\t'
        else:
            sep = ','
        geo_lookup = pd.read_csv(state_path + geo_file, sep=sep,
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
        logging.error('unable to process geo file in {tf} for {y}'.\
                      format(tf=template_folder, y=year))
        logging.error(str(ex))



def build_raw_file(state, year, col_lookup, geo_lookup, state_path=None, 
        check_types=False, max_vars=2000, output_path='acs_{year}_output/'):
    """ download and combine acs files for a state into one raw csv file

    :param state: string state name (full name)
    :param year: int four-digit year
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
        default: acs_{y}_output/
        where y=year

    :return: updated col_lookup dataframe
    """

    logging.info('building raw files for {0}'.format(state))

    std_cols = ['ignore', 'acs_type', 'state', 'ignore2', 'file_num', 
        'logrecno']
    std_types = {'ignore':'str', 'acs_type':'str', 'state':'str', 
        'ignore2':'int', 'file_num':'int', 'logrecno':'int'}

    output_path = output_path.format(year=year)

    if not state_path:
        state_path = get_state_data(state, year)

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
                               index_col=False, na_values='.', 
                               encoding='latin-1')
            for c in data_cols:
                if re.match('^MEDIAN .*', str(labels[c])) or \
                    re.match('^AGGREGATE .*', str(labels[c])):
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
        col_lu_file = '{0}/col_lookup.csv'.format(output_path)
        col_lookup.to_csv(col_lu_file, index=False)
        logging.info('updated column lookup written to col_lookup.csv')


def prep_acs_main(state, year, template_folder=None, state_path=None, 
        check_types=False, max_vars=2000, output_path='acs_{year}_output/'):
    """ prep ACS data

    :param state: string state name (full name)
    :param year: int four-digit year
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
        default: acs_{year}_output/

    :return: None
    """

    logging.basicConfig(format='%(asctime)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    num_logger = logging.getLogger('numexpr')
    num_logger.setLevel(logging.ERROR)

    logging.info('start of prep_acs for {st} in {y}'.format(st=state, y=year))

    output_path = output_path.format(year=year)

    # make sure output path exists
    temp_path = './{op}'.format(op=output_path)
    pathlib.Path(temp_path).mkdir(parents=True, exist_ok=True)

    # get column lookup
    if check_types:
        col_lu, template_folder = build_column_lookup(year, template_folder)
        if col_lu is None:
            return
    else:
        col_lu_file = '{0}/col_lookup.csv'.format(output_path)
        col_lu = pd.read_csv(col_lu_file)
        if os.path.isdir('{0}/{1}'.format(template_folder, 'templates')):
            template_folder = '{0}/{1}'.format(template_folder, 'templates')

    # get state data
    if not state_path:
        state_path = get_state_data(state, year)
        if not state_path:
            return

    # set up geoids
    logging.info('set up geoids')
    geo_lu = build_geo_lookup(state_path, template_folder, year)
    if geo_lu is None:
        return
    elif geo_lu.shape[0] == 0:
        logging.error('no geoids read for {s} in {y}'.format(s=state, y=year))
        return

    # process raw data
    col_lu = build_raw_file(state, year, col_lu, geo_lu, state_path=state_path, 
        check_types=check_types, max_vars=max_vars, 
        output_path=output_path)

    # clean up raw state files
    logging.info('cleaning up raw files')
    os.remove('{st}.zip'.format(st=state))
    shutil.rmtree(state)

    logging.info('prep_acs completed for {st} in {y}'.format(st=state, y=year))


def prep_acs_full(year, max_vars=2000, output_path='acs_{year}_output/'):
    """ download and prep all ACS data for a given year

    Convenience method to download all data for a given year.  This will take
    multiple hours to run.

    :param year: int four-digit year
    :param max_vars: number of variables to output per file
        default: 2000
        there are over 22,000 variables total which takes hours to export as one
            big file
    :param output_path: string path to write raw files to
        default: acs_{year}_output/

    :return: None
    """

    # start with last state, Wyoming, since its small
    state = STATE_NAMES.pop()
    prep_acs_main(state, year, template_folder=None, 
        state_path=None, check_types=True, max_vars=max_vars, 
        output_path=output_path)    

    for state in STATE_NAMES:
        prep_acs_main(state, year, template_folder='templates_{}'.format(year), 
            state_path=None, check_types=False, max_vars=max_vars, 
            output_path=output_path)    

    return


def prep_acs_cmd():

    parser = argparse.ArgumentParser(description='Prep ACS data')
    parser.add_argument('year', type=int, help='four digit year')
    parser.add_argument('state', type=str, help='state full name', nargs='+')

    parser.add_argument('-a', '--all', default=False, action='store_true',
        help='download all data for a given year.  '
             'only max_vars and output_path parameters are applied')
    parser.add_argument('-tf', '--template_folder', default=None,
        help='template folder path. if None, will download to templates/')
    parser.add_argument('-sp', '--state_path', default=None,
        help='raw state data path. if None, will download to state folder')
    parser.add_argument('-ct', '--check_types', default=False, 
        action='store_true', help='check data types on load')
    parser.add_argument('-mv', '--max_vars', default=2000, type=int,
        help='approximate number of variables to output per file. '  
        'the script allows up to 10 pct more to keep the number of files down')
    parser.add_argument('-op', '--output_path', default='acs_{year}_output/',
        help='path to write raw files to')

    args = parser.parse_args()

    if args.all:
        prep_acs_full(args.year, max_vars=args.max_vars, 
                      output_path=args.output_path)
    else:
        for st in args.state:
            prep_acs_main(st, args.year, template_folder=args.template_folder, 
                state_path=args.state_path, check_types=args.check_types,
                max_vars=args.max_vars, output_path=args.output_path)


if __name__ == '__main__':
    prep_acs_cmd()

