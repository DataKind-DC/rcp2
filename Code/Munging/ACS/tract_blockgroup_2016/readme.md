# Prep ACS Tract & Blockgroup Features

## Introduction
This folder contains two scripts: one that downloads raw ACS data, and<br>
one that builds standardized features from that raw data.

## Setup
* Make sure the necessary Python packages are installed using <br>
`pip install -r requirements.txt`

## Sample Workflow
```
# download data
python prep_acs_tract_block.py 2012 Alabama --check_types
python prep_acs_tract_block.py 2012 Alaska Arkansas  --template_folder 'templates_2012/'

# build features
python build_acs_features.py 2012 acs_munging.txt
```

-----

## Download Raw Data
The `prep_acs_tract_block.py` script downloads raw ACS 5-year<br>
data at the Census tract and blockgroup level for the specified states in the<br>
given year.  The state name inputs should be the full name, in camel-case, <br>
with spaces removed.  For example, `Arizona` and `NewHampshire` are valid while <br>
other variations are not.

The data is downloaded from<br>
https://www2.census.gov/programs-surveys/acs/summary_file/2016/data/5_year_by_state/ <br>
with appropriate year inserted (2016 in this example).<br>
The associated templates are pulled from <br>
https://www2.census.gov/programs-surveys/acs/summary_file/2016/data/2016_5yr_Summary_FileTemplates.zip

### Usage
You should only need to download the templates, and check the types, once.

For the first state:<br>
`python prep_acs_tract_block.py Alabama --check_types` <br>
This will output col_lookup.csv in the same folder as the data, that is <br>
ingested by future calls to the script without the `--check_types` option.

For subsequent states:<br>
`python prep_acs_tract_block.py Alaska Arkansas --template_folder 'templates_{year}/'`

### Full Specification
```
usage: prep_acs_tract_block.py [-h] [-tf TEMPLATE_FOLDER] [-sp STATE_PATH]
                               [-ct] [-mv MAX_VARS] [-op OUTPUT_PATH]
                               year state [state ...]

Prep ACS data

positional arguments:
  year                  four digit year
  state                 state full name

optional arguments:
  -h, --help            show this help message and exit
  -tf TEMPLATE_FOLDER, --template_folder TEMPLATE_FOLDER
                        template folder path. if None, will download to
                        templates/
  -sp STATE_PATH, --state_path STATE_PATH
                        raw state data path. if None, will download to state
                        folder
  -ct, --check_types    check data types on load
  -mv MAX_VARS, --max_vars MAX_VARS
                        approximate number of variables to output per file.
                        the script allows up to 10 pct more to keep the number
                        of files down
  -op OUTPUT_PATH, --output_path OUTPUT_PATH
                        path to write raw files to
```

-----

## Build Features
The `build_acs_features.py` script pulls the variables of interest out of<br>
the raw files downloaded by the above script and performs the specified <br>
transformations.

### Usage
The first task to is build the set of transformations.  This should be a tab<br>
delimited text file with columns: variable_name, operator, argument1, argument2<br>
Basic arithmetic operators are supported: +, -, *, /.  Arguments can be column<br>
names or numeric values.  Alternatively, you can include a list of variables,
one per row, under the variable_name header to select raw variables without
any recoding.  The list of column names can be found in the column lookup file<br>
built by `prep_acs_tract_block.py`.

An example transform file is included here as `acs_munging.txt`.

Then run:<br>
`python build_acs_features.py year transform_file`

### Full Specification
```
usage: build_acs_features.py [-h] [-lu LOOKUP_FILE] [-afp ACS_FILES_PATH]
                             [-o OUTPUT_FILE]
                             year vars_file

Build ACS features

positional arguments:
  year                  four digit year
  vars_file             file with list of vars or list of transformations

optional arguments:
  -h, --help            show this help message and exit
  -lu LOOKUP_FILE, --lookup_file LOOKUP_FILE
                        --column lookup filename (default
                        acs_{year}_output/col_lookup.csv)
  -afp ACS_FILES_PATH, --acs_files_path ACS_FILES_PATH
                        folder with raw data (default acs_{year}_output)
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        output file name (default acs_{year}_features)
```
