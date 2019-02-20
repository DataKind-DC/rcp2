Overall Objective | Road Map | How to Help and Get Involved | Input Data

# American Red Cross Campaign
What areas in the USA are at the highest risk of home fires and where should the American Red Cross go to install smoke alarms?

The American Red Cross (RC) Home Fire Prevention Campaign began in 2014 and is geared towards educating the general public about fire safety and providing tools such as smoke alarms to improve the preparedness of an area to home fires. So far, ARC has installed over 1 million smoke alarms across the USA.

In 2015, DataKind DC, the American Red Cross, and Enigma worked together to create a Home-Fire Risk Score at the U.S. Census Tract Level. The team created models to predict the highest impact areas to go knock on doors to install smoke alarms. These models were developed using proprietary data from the American Red Cross, the American Community Survey, the American Housing Survey, and NFIRS. Since the Red Cross doesn't visit all areas of the USA, risk scores were imputed to non-surveyed areas. Results were displayed on the smoke signals map. Some work was also done to understand historical home fires, where did it happen, how deadly were they?

The final product was a Home Fire Risk <a href="http://www.datakind.org/blog/american-red-cross-and-datakind-team-up-to-prevent-home-fire-deaths-and-injuries">Map</a>.

Phase 1's original GitHub repo is <a href="https://github.com/DataKind-DC/smoke_alarm_models">here</a>.

## Project objective
Create a fire risk map to identify areas that are least prepared to respond to house fires. Risk will be measured by:
Predicting where fires occur
Predicting the maximum likelihood of fire deaths that would occur




## Project Road Map
1. Work in monthly sprints
2. Document input datasets.  We have new NFIRS, Red Cross, and ACS data that we would like to incorporate. We would like to consider adding new types of data as well such as climate data.
3. Create a risk model.
4. Update the Home Fire Risk Map!

## How to help out
1. Check out Projects board
2. Check out the Wiki for more background information
3. Message the team coordinators on DKDC slack @Sherika, @matt.barger, and @judy

## How to get Involved
1. Attend DKDC DataJams to meet the team and learn about the project.  
2. If you are interested in being more involved, join our RC team work nights
3. Feel free to contribute to our GitHub anytime


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
