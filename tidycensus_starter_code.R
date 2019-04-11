#Intro to 'tidycensus' package
#Matt Barger

rm(list = ls())
library(tidyverse)
library(tidycensus)
#'tidycensus' is probably the best R package for interacting with the Census API.
# More information here: https://walkerke.github.io/tidycensus/

#To find all variables, run the load_variables() function to grab a searchable list of all available variables
acs5Var <- load_variables(2016, "acs5")

#NB: An easy way to isolate all states for the upcoming API pull is to isolate unique abbreviations from the FIPS code database
all_states <- unique(fips_codes$state)[1:51]

#To extract data itself, use the get_acs() function as follows 
pop_by_tract <- get_acs(geography = "tract",
                        variables = "B01003_001",
                        state = all_states)

write.csv(pop_by_tract, "pop_tract.csv", row.names = F)


