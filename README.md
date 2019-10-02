Overall Objective | Road Map | How to Help and Get Involved | Input Data

# American Red Cross Campaign
What areas in the USA are at the highest risk of home fires, and where should the American Red Cross go to install smoke alarms?

In Phase 1 DKDC created 6 models to analyze fire response data, smoke alarm data, and census data to assign a fire risk score to census tracts across the United States. The results from these models helped generate a map of high-risk census tracts across the United States, which informed planning and helped us adjudicate resources. Now, DKDC has been asked to replicate this effort at a census block level (a smaller geographic unit), so that the Red Cross can more efficiently target smoke detector distribution efforts. This phase of work will help ensure fire alarms are handed out where they are most needed.
 
Phase 2 has three primary objectives:
1.	Refine and update risk model to include smaller geographic areas and new data.
2.	Update the user interface so Red Cross end users can interact with the risk scores, view their more specific components, and prioritize locations to distribute smoke detectors.
3.	Set up a method so that the model can easily be refreshed by the Red Cross team when new home fire datasets are available.

## Project Discussion and Materials
Project Discussion is on the DKDC Slack in the #rcp2_public channel.

Data Location and Dictionaries: We have new NFIRS, Red Cross, and ACS data that we would like to incorporate. We would like to consider adding new types of data as well such as climate data. The data can be found here (link to google drive?).

Phase 1 Map: <a href="http://home-fire-risk.github.io/smoke_alarm_map/">Fire Risk Map</a>

Phase 1 Blog Post: <a href="http://www.datakind.org/blog/american-red-cross-and-datakind-team-up-to-prevent-home-fire-deaths-and-injuries">DataKind Blog</a>.

## How to Get Involved and Help Out
Please review the skills we are looking for below, and let us know if you’d like to get involved by emailing a data ambassador or posting in the Slack channel - we’d love your help!

Skills used/needed: 
There are two main components of the project: data modeling, and visualization. The modeling part requires aggregating, joining, and geocoding large datasets, and modeling fire risk from the variables contained. R has been the main language for the project so far and is preferred, but python or other data analysis welcome also. The visualization portion of the project needs front-end web development skills, particularly Mapbox GL, D3.js, html and javascript.

More reading and background: 
The GitHub Repo contains links to the onboarding powerpoint, notes on the original model, and related background research.

Current To Dos: 
Right now, we need help downloading and pre-processing U.S. census data at the block and/or block group level, and merging this with the home fire and other data. After that, we will need help creating home fire risk score machine learning models and updating the user interface. Specific tasks can be found in the GitHub Project Board and Issues.

## Input Data Sources


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
