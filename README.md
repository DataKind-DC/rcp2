

## Background

What areas in the USA are at the highest risk of home fires and where should the American Red Cross go to install smoke alarms?

In 2015, DataKind DC, the American Red Cross, and Enigma worked together to create a Home-Fire Risk Score at the U.S. Census Tract Level. The team created models to predict the highest impact areas to go knock on doors to install smoke alarms. These models were developed using proprietary data from the American Red Cross, the American Community Survey, the American Housing Survey, and NFIRS. Since the Red Cross doesn't visit all areas of the USA, risk scores were imputed to non-surveyed areas. Results were displayed on the smoke signals map. Some work was also done to understand historical home fires, where did it happen, how deadly were they?

The final product was a Home Fire Risk <a href="http://www.datakind.org/blog/american-red-cross-and-datakind-team-up-to-prevent-home-fire-deaths-and-injuries">Map</a>.

Phase 1's original GitHub repo is <a href="https://github.com/DataKind-DC/smoke_alarm_models">here</a>.


## Project Road Map
1. Work in monthly sprints
2. Document input datasets.  We have new NFIRS, Red Cross, and ACS data that we would like to incorporate. We would like to consider adding new types of data as well such as climate data.
3. Create a risk model.
4. Update the Home Fire Risk Map!

## How to help out
1. Check our project and issue board
2. Message the team coordinators on DKDC slack @Sherika, @matt.barger, and @judy

## How to get Involved
1. Attend DKDC DataJams to meet the team and learn about the project.  
2. If you are interested in being more involved, join our RC team work nights
3. Feel free to contribute to our GitHub anytime


## Model Input-Output Relationships


Input Data | Output Files | Output To...
-------|---------------|-------------
Homefire_SmokeAlarmInstalls.csv, ACS | smoke_alarm_risk_scores_1a.csv  | Aggregate
2009_2013_alarm_tract_data.csv, ACS  | tracts_74k_weighted_linear_preds_upsampled.csv | Aggregate
AHS  | smoke_alarm_risk_scores_1c.csv  | Aggregate
2010 Census, fire_incident  | 2009_tract_building_fire_per_1k 2010_tract_building_fire_per_1k 2011_tract_building_fire_per_1k 2012_tract_building_fire_per_1k 2013_tract_building_fire_per_1k   | Aggregate
RC disaster cases, 2010 census tract shape files, state fibs code file  | fires_per_tract  | Aggregate
modeling_injury_dataset, ACS tract data  | results_tract  | Aggregate

## DataKind DataCorps

DataKind DataCorps brings together teams of pro bono data scientists with social change organizations on long-term projects that use data science to transform their work and their sector. We help organizations define their needs and discover what’s possible, then match them with a team that can translate those needs into data science problems and solve them with advanced analytics.

We are very proud to re-partner with the American Red Cross!
