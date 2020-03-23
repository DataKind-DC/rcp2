# Prep ACS Tract & Blockgroup Features

## Introduction
This folder contains two scripts: one that downloads raw ACS data, and<br>
one that builds standardized features from that raw data.

## Setup
* Make sure the necessary Python packages are installed using <br>
```pip install -r requirements.txt```

## Download Raw Data
The ```prep_acs_tract_block.py``` script downloads raw 2016 ACS 5-year<br>
data at the Census tract and blockgroup level for the specified states.<br>
The state name inputs should be the full name, in camel-case, with spaces <br>
removed.  For example, ```Arizona``` and ```NewHampshire``` are valid while <br>
other variations are not.

The data is downloaded from<br>
https://www2.census.gov/programs-surveys/acs/summary_file/2016/data/5_year_by_state/ <br>
The templates are pulled from <br>
https://www2.census.gov/programs-surveys/acs/summary_file/2016/data/2016_5yr_Summary_FileTemplates.zip

### Usage
You should only need to download the templates, and check the types, once.

For the first state:<br>
```python prep_acs_tract_block.py Alabama --check_types``` <br>
This will output col_lookup.csv in the current working directory that is <br>
ingested by future calls to the script without the ```--check_types``` option.

For subsequent states:<br>
```python prep_acs_tract_block.py Alaska Arkansas --template_folder 'templates/'```

### Full Specification
```
usage: prep_acs_tract_block.py [-h] [-tf TEMPLATE_FOLDER] [-sp STATE_PATH]
                               [-ct] [-mv MAX_VARS] [-op OUTPUT_PATH]
                               state [state ...]

Prep ACS data

positional arguments:
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

## Build Features
The ```build_acs_features.py``` script pulls the variables of interest out of<br>
the raw files downloaded by the above script and divides them by the total <br>
variables for use in modeling.

### Usage
The first task to is build a list of variable names, one per line, from the<br>
 column lookup file built by ```prep_acs_tract_block.py```.

Then run:<br>
```python build_acs_features.py vars_file lookup_file acs_files_path```

### Full Specification
```
usage: build_acs_features.py [-h] [-o OUTPUT_FILE]
                             vars_file lookup_file acs_files_path

Build ACS features

positional arguments:
  vars_file             file with list of vars
  lookup_file           column lookup filename
  acs_files_path        folders with raw data

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        output file name (default acs_features)
```
