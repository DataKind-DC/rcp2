


# Red Cross Fire Risk Map v2
What areas in the USA are at the highest risk of home fires, and where should the American Red Cross go to install smoke alarms?

In Phase 1 DKDC created 6 models to analyze fire response data, smoke alarm data, and census data to assign a fire risk score to census tracts across the United States. The results from these models helped generate a map of high-risk census tracts across the United States, which informed planning and helped us adjudicate resources. Now, DKDC has been asked to replicate this effort at a census block level (a smaller geographic unit), so that the Red Cross can more efficiently target smoke detector distribution efforts. This phase of work will help ensure fire alarms are handed out where they are most needed.
 
Phase 2 has three primary objectives:
1.	Refine and update risk model to include smaller geographic areas and new data.
2.	Set up a method so that the model can easily be refreshed by the Red Cross team when new home fire datasets are available.

## Quickstart Guide

#### 1. Get on our Slack Channel 
 https://dkdc.herokuapp.com/
#### 2. Get the data repository link from RCP2_public
#### 3. Download "Master Project Data" folder   RCP2 > 02_data > Master Project Data 
#### 4. Fork this repo and place 'Master Project Data' into Data folder 

#### 5. Python installation (optional but recommended)
Download anaconda ( https://anaconda.org/) 

go to command line ( or anaconda terminal) and navigate to this directory (usually documents/github/rcp2)
  ```
  conda create -n RCP2 python=3 -c conda-forge
  conda activate RCP2
  conda install --file requirements.txt
  ```
  This will create your environment and activate and you'll be ready to go. 

#### 6. (optional) download github desktop
make your life easier if you are new to github
https://desktop.github.com/

#### 7. (optional) Read Up
in the Google Drive there is a lot of great resources compiled in both Master Project Data and on the drive at  01_project_overview > Additional reading 

#### 8. Find a Task
Click on the Projects board (above) and then RCP2 to get a look at all the current tasks


## Project Discussion and Materials
Project Discussion is on the DKDC Slack in the #rcp2_public channel.

Data Location and Dictionaries: We have new NFIRS, Red Cross, and ACS data that we would like to incorporate. We would like to consider adding new types of data as well such as climate data. 

Phase 1 Map: <a href="http://home-fire-risk.github.io/smoke_alarm_map/">Fire Risk Map</a>

Phase 1 Blog Post: <a href="http://www.datakind.org/blog/american-red-cross-and-datakind-team-up-to-prevent-home-fire-deaths-and-injuries">DataKind Blog</a>.

## How to Get Involved and Help Out
Please review the skills we are looking for below, and let us know if you’d like to get involved by emailing a data ambassador or posting in the Slack channel - we’d love your help!

Skills used/needed: 
There are two main components of the project: data modeling, and visualization. The modeling part requires aggregating, joining, and geocoding large datasets, and modeling fire risk from the variables contained. Python been the main language for the project so far and is reccomended for beginners, but R,tableau,GIS is also welcome if you are more comfortable with using it. The visualization portion of the project needs front-end web development skills, particularly Mapbox GL, D3.js, html and javascript.

Input Data | Folder Name | Geo Type |  Description / Comments
-------|-----------|-------------|-------------
American Community Survey | 02_inputdata_ACS | census tract | socio-economic variables
American Housing Survey | | |
American Red Cross Preparedness Data | | | ARC home visits for smoke alarm installation and fire safety instruction. Includes the number of smoke alarms installed, environmental hazards, the number of alarms that existed in a dwelling prior to a ARC visit, etc.
American Red Cross Response Data  | | |ARC home visits for smoke alarm installation and fire safety instruction. Includes the number of smoke alarms installed, environmental hazards, the number of alarms that existed in a dwelling prior to a ARC visit, etc.
American Red Cross Boundary Data | | ARC region/chapters, zip codes|
Census  | | |
Homeland Infrastructure Foundation Level Data (HIFLD) fire station locations  | | | 2010 list of >50k fire stations and their latitude & longitude coordinates in the USA [source]. 2017 Census tract & blocks added. Shapefiles can be found at source.
HIFLD emergency medical service locations  | | | 2010 list of ambulatory and EMS locations in the USA [source]. 2017 Census tract & blocks added. Shapefiles can be found at source.
NFIRS  | | Census tract | CSV-file that contains the address, latitude & longitude, Census tract, and Census block information for home fires in the USA from 2009-2016
SVI  | | | 2016 CDC’s Social Vulnerability Index which is based off of Census’s American Community Survey data. Includes indexes for socioeconomic, household composition and disability, minority status and language, and housing and transportation that summarizes a population’s resilience to stressors.

## DataKind DataCorps

DataKind DataCorps brings together teams of pro bono data scientists with social change organizations on long-term projects that use data science to transform their work and their sector. We help organizations define their needs and discover what’s possible, then match them with a team that can translate those needs into data science problems and solve them with advanced analytics.

We are very proud to re-partner with the American Red Cross!


Project Organization
------------

    ├── LICENSE
    |.  requirements.txt   <- list of python packages currently used in project. 
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├──
    │   ├── interim        <- Intermediate data that you have transformed. 
    │   ├── Master Project Data <- The final, canonical data sets for modeling.(on google drive)
    │   └── raw            <- The original, immutable data dump. ( on google drive)
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials (moved to google drive in Master                                                                                                project data).
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    |.  ----(not currently implemented) ---- 
    |          Future roadmap
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
=======



