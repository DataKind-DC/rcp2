#########
#INITIAL RUN OF MASTER DATASET COMPILING
## BY MATT BARGER (Hi, I'm Matt.)
##
#Notes: 
# Summary: I'm using a simple bind_rows function to put all the datasets together, but you want to make sure that two unifiers (GEOID and date_year) are aligned.
#
# 0. Obviously this isn't perfect. Changes welcome. Specifically:
# 0a. add date and time increments for NFIRS and ARC_Preparedness database
# 0b. add process to build on ACS/AHS/SVI data
# 0c. add instructions for non-time-based data (i.e., ARC_saved_lives data)
#
# 1. All data comes from the RCP2 data drive. Your filenames may be different from mine
# 2. The files are big. Prepare for that. You will need more than 4 GB of RAM.

rm(list = ls())

library(tidyverse)
library(tidycensus)
library(readr)


##NFIRS DATA COMBINED BY CENSUS BLOCK GROUP (obv find latest file)... don't worry about the error message (it's for the time variable)
NFIRS_2009_2016_Combined_Census_Tract_1_ <- read_csv("NFIRS_2009_2016_Combined_Census_Tract (1).csv")
nfirs <- NFIRS_2009_2016_Combined_Census_Tract_1_
nfirs <- nfirs %>% 
  mutate(data_id = "NFIRS") %>%
  separate(inc_date,
           into = c("inc_date", "inc_time"),
           sep = " ") %>%
  mutate(date_year = substr(inc_date, -4, -1),
         GEOID = as.character.numeric_version(GEOID))
  
##ARC RESPONSE DATA
ARC_RESPONSE <- read_csv("ARC_RESPONSE.csv", 
                         col_types = cols(Zip = col_character(), 
                                          FIPS = col_character()))
library(tidyverse)
arc <- ARC_RESPONSE %>% 
  mutate(data_id = "ARC-RESPONSE", date_year = as.character(Year)) %>% 
  select(GEOID = FIPS, everything())

##ARC SAVED LIVES DATA, No date or time
ARC_SavedLives <- read_csv("ARC_SavedLives.csv", col_types = cols(Zip = col_character(), FIPS = col_character()))
arc_sl <- ARC_SavedLives %>% 
  mutate(data_id = "ARC-SAVEDLIVES") %>%
  select(GEOID = FIPS, everything())

##ARC PREPAREDNESS DATA, All date and time
ARC_Prep <- read_csv("ARC_Prep.csv", col_types = cols(Zip = col_character(),GEOID = col_character()))
arc_p <- ARC_Prep %>% 
  mutate(data_id = "ARC-PREP", date_date = as.Date(`In-Home Visit Date`, origin = "1899-12-30")) %>%
  mutate(date_year = as.character(format(date_date, '%Y')),
         GEOID = as.character(FIPS))

#JOIN DATA, WRITE TO CSV
nfirs_arc <- bind_rows(arc, arc_sl, arc_p, nfirs)

### THIS IS WHERE I RAN OUT OF RAM :( :( :(, so, points moving forward:
# GATHER name-of-region data from tidycensus
# GET NAME OF REGION DATA
all_states <- unique(fips_codes$state)[1:51]

#To extract data itself, use the get_acs() function as follows 
names_of_regions_by_tract <- get_acs(geography = "tract",
                                     variables = "B01003_001",
                                     state = all_states) %>% select(1:4)

## THEN SEPARATE THE NAME DATA BY ", " to get TRACT, COUNTY, STATE names by GEOID. 

## We... did a lot of stuff by this point. You might want to write to csv HERE.
write.csv(nfirs_arc, "nfirs_arc.csv", row.names = F)


#THEN, FOR SAMPLE ANALYSIS:
## 1. GROUP BY date_year and STATE, COUNTY, TRACT
## 2. SUMMARIZE by counts (i.e., 'n()') or by sums for variables within the NFIRS data


