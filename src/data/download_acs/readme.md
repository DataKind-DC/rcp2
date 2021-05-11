# Download ACS Data

## Introduction
This folder contains one script that downloads ACS data from the Census API.
<br>
It has been tested on ACS 5-year estimates in 2014, 2015, 2016, 2018.

## Setup
* Make sure the necessary Python packages are installed using <br>
`pip install -r requirements.txt`

## Sample Workflow
```
# download data and build features
python acs_blockgroup.py 2016 acs_2016_munging.txt all --all
```

-----

## Download Data and Build Features
The `acs_blockgroup.py` script queries the Census API for ACS 5-year<br>
data at blockgroup level for the vars in the states in the<br>
given year. It then performs the specified <br>
transformations.

The data is downloaded from https://api.census.gov/data/2016/acs/acs5.<br>
with appropriate year inserted (2016 in this example).<br>


### Usage
Use `python acs_blockgroup.py {year} {munging file} all --all` with the appropriate year <br> to download data for all states in a give year.<br>
Example: `python acs_blockgroup.py 2016 acs_2016_munging.txt all --all`

If you want to run select states only, use:<br>

`python prep_acs_blockgroup.py {year} {munging file} {state}` <br>
Example: `python acs_blockgroup.py 2016 acs_2016_munging.txt Wyoming` <br>

For multiple states:
`python prep_acs_blockgroup.py {year} {munging file} {state1} {state2}` <br>
Example: `python acs_blockgroup.py 2016 acs_2016_munging.txt Wyoming Alaska` <br>


After the data is downloaded, transformations specified in the munging file<br>
will be performed on it. The munging file should be a tab<br>
delimited text file with columns: variable_name, operator, argument1, argument2<br>
Basic arithmetic operators are supported: +, -, *, /.  Arguments can be column<br>
names or numeric values.  Alternatively, you can include a list of variables,
one per row, under the variable_name header to select raw variables without
any recoding.



### Full Specification
```
usage: acs_blockgroup.py [-h] [-a] [-cs] [-op OUTPUT_PATH] year vars_file state [state ...]

Download ACS data and build features

positional arguments:
  year                  four digit year
  vars_file             file with list of vars or list of transformations
  state                 state full name

optional arguments:
  -h, --help            show this help message and exit
  -a, --all             download data for all states for a given year. only output_path parameters are applied
  -cs, --check_states   check if states have already been downloaded
  -op OUTPUT_PATH, --output_path OUTPUT_PATH
                        path to write files to
```