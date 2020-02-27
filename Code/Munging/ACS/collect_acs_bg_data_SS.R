---
title: "Gather Census Block Group Data"
author: "Matt Wibbenmeyer"
date: "February 8, 2020"
output: html_document
---
  
  
##### Edited by Sumati Sridhar, 02-20-2020 #####  
  
  
#This script extracts ACS 5-year estimates at the block group (or any larger 
#geography) using the tidycensus package. To run tidycensus, you first need
#to set up a Census API key and run census_api_key(). Set working directory
#to where you want output files to save, or use the collect_acs_data function 
#to set a different outpath.



if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, tidycensus, viridis,stringr,dplyr,knitr,DT,datasets)

#For code to run, need to first set up Census API key and run census_api_key()

acs_table <- load_variables(2016, "acs5", cache = TRUE)
view(acs_table)

#Set ACS parameters
# geography = 'block group'
# year = 2016
# state = 'OR'
# county = 'Lane'

collect_acs_data <- function(geography = 'tract', 
                             year = 2018, 
                             state = NULL, 
                             county = NULL,
                             outpath = '') {
  
    print(paste0('Processing state: ',state))
  
    #Income
    print(paste0('Extracting income variables....'))
    income <- get_acs(geography = geography, sumfile = 'acs5', 
                      variables = c('B19301_001', 'B17021_001', 'B17021_002'),
                      year = year, state = state, county = county, geometry = FALSE) %>%
              select(-moe)%>%
              spread(key = 'variable', value = 'estimate') %>%
              mutate(
                tot_population = B17021_001,
                in_poverty = B17021_002) %>%
              mutate(
                inc_pct_poverty = in_poverty/tot_population,
                inc_pcincome = B19301_001
              ) %>%
              select(-starts_with("B1"))
    
    #Race
    print(paste0('Extracting race variables....'))
    race <- get_acs(geography = geography, sumfile = 'acs5',
                    variables = c(sapply(seq(1,10,1), function(v) return(paste("B02001_",str_pad(v,3,pad ="0"),sep=""))),
                                  'B03002_001','B03002_002','B03002_003','B03002_012','B03002_013'),
                    year = year, state = state,county = county, geometry = FALSE) %>%
                    select(-moe) %>%
                    spread(key = 'variable', value = 'estimate') %>% 
                    mutate(
                      race_pct_white = B02001_002/B02001_001,
                      race_pct_whitenh = B03002_003/B03002_001,
                      race_pct_nonwhite = 1 - race_pct_white,
                      race_pct_nonwhitenh = 1 - race_pct_whitenh,
                      race_pct_amind = B02001_004/B02001_001,
                      race_pct_black = B02001_003/B02001_001,
                      race_pct_hisp = B03002_012/B03002_001
                    ) %>%
                    select(-starts_with("B0"))  
    
    #Age
    print(paste0('Extracting age variables....'))
    age <- get_acs(geography = geography, sumfile = 'acs5',
                   variables = c(sapply(seq(1,49,1), function(v) return(paste("B01001_",str_pad(v,3,pad ="0"),sep="")))),
                   year = year, state = state, county = county, geometry = FALSE)%>%
                select(-moe) %>%
                spread(key = 'variable', value = 'estimate') %>% 
                mutate(
                  denom = B01001_001,
                  age_under25_ma = select(., B01001_003:B01001_010) %>% rowSums(na.rm = TRUE),
                  age_25_64_ma = select(., B01001_011:B01001_019) %>% rowSums(na.rm = TRUE),
                  age_over65_ma = select(., B01001_020:B01001_025) %>% rowSums(na.rm = TRUE),
                  age_under25_fe = select(., B01001_026:B01001_034) %>% rowSums(na.rm = TRUE),
                  age_25_64_fe = select(., B01001_035:B01001_043) %>% rowSums(na.rm = TRUE),
                  age_over65_fe = select(., B01001_044:B01001_049) %>% rowSums(na.rm = TRUE),
                  age_pct_under25 = (age_under25_ma + age_under25_fe)/denom,
                  age_pct_25_64 = (age_25_64_ma + age_25_64_fe)/denom,
                  age_pct_over65 = (age_over65_ma + age_over65_fe)/denom
                ) %>%
                select(-starts_with("B0")) %>% select(-ends_with("_ma")) %>% select(-ends_with("_fe")) %>% select(-denom)
    
    #Education
    print(paste0('Extracting education variables....'))
    educ <- get_acs(geography = geography, sumfile = 'acs5',
                    variables = c(sapply(seq(1,35,1), 
                                         function(v) return(paste("B15002_",str_pad(v,3,pad ="0"),sep="")))),
                    year = year, state = state, county = county, geometry = FALSE) %>%        
      select(-moe) %>% spread(key = 'variable', value = 'estimate') %>%
      mutate(educ_tot_pop = B15002_001, 
             educ_no_school = (select(., B15002_003, B15002_020) %>% rowSums(na.rm = T))/educ_tot_pop, 
             educ_nursery_4th = (select(., B15002_004, B15002_021)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_5th_6th = (select(., B15002_005, B15002_022)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_7th_8th = (select(., B15002_006, B15002_023)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_9th = (select(., B15002_007, B15002_024)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_10th = (select(., B15002_008, B15002_025)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_11th = (select(., B15002_009, B15002_026)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_12th_no_diploma = (select(., B15002_010, B15002_027)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_high_school_grad = (select(., B15002_011, B15002_028)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_col_less_1_yr = (select(., B15002_012, B15002_029)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_some_col_no_grad = (select(., B15002_013, B15002_030)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_associates = (select(., B15002_014, B15002_031)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_bachelors = (select(., B15002_015, B15002_032)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_masters = (select(., B15002_016, B15002_033)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_professional = (select(., B15002_017, B15002_034)%>% rowSums(na.rm = T))/educ_tot_pop,
             educ_docterate = (select(., B15002_018, B15002_035)%>% rowSums(na.rm = T))/educ_tot_pop
      ) %>%
      select(-starts_with("B1"))
    
    
    # work status
    print(paste0("Extracting employment status variables..."))
    work = get_acs(geography = geography, sumfile = 'acs5',
                   variables = c(sapply(seq(1,36,1), 
                                        function(v) return(paste("B23027_",str_pad(v,3,pad ="0"),sep="")))),
                   year = year, state = state, county = county, geometry = FALSE) %>% 
      
      select(-moe) %>% spread(key = 'variable', value = 'estimate') %>%
      mutate(total_pop_16_plus = B23027_001, 
             worked_past_12_mo_sum = select(., B23027_003, B23027_008, B23027_013, B23027_018, B23027_023, B23027_028, B23027_033) %>% rowSums(na.rm = TRUE),
             worked_past_12_mo = worked_past_12_mo_sum/total_pop_16_plus,
             did_not_work_past_12_mo_sum = select(., B23027_006, B23027_011, B23027_016, B23027_021, B23027_026, B23027_031, B23027_036) %>% rowSums(na.rm = T),
             did_not_work_past_12_mo = did_not_work_past_12_mo_sum / total_pop_16_plus
      ) %>%
      select(-worked_past_12_mo_sum, -did_not_work_past_12_mo_sum, -starts_with('B2'))
    
    
    #Household composition 
    print(paste0('Extracting household composition variables....'))
    house = get_acs(geography = geography, sumfile = 'acs5',
                    variables = c(sapply(seq(1,9,1), function(v) return(paste("B11001_",str_pad(v,3,pad ="0"),sep="")))),
                    year = year, state = state, county = county, geometry = F) %>% 
      select(-moe) %>%  
      spread(key = 'variable', value = 'estimate') %>%
      mutate(house_tot_occ_cnt = B11001_001,
             house_pct_family = (B11001_002/house_tot_occ_cnt),
             house_pct_family_married = B11001_003/house_tot_occ_cnt, 
             house_pct_family_male_hh = B11001_005/house_tot_occ_cnt, 
             house_pct_family_female_hh = B11001_006/house_tot_occ_cnt, 
             house_pct_non_family = B11001_007/house_tot_occ_cnt, 
             house_pct_live_alone = B11001_008/house_tot_occ_cnt, 
             house_pct_no_live_alone = B11001_009/house_tot_occ_cnt) %>%
      select(-starts_with('B1'))
    
    #Houshold units and occupancy
    print(paste0('Extracting household occupancy variables...'))
    occupancy = get_acs(geography = geography, sumfile = 'acs5',
                        variables = c(sapply(seq(1,3,1), function(v) return(paste("B25002_",str_pad(v,3,pad ="0"),sep="")))),
                        year = year, state = state, county = county, geometry = F) %>% 
      select(-moe) %>%
      spread(key = 'variable', value = 'estimate') %>% 
      mutate(total_housing_units = B25002_001, 
             house_pct_occupied = B25002_002 / total_housing_units, 
             house_pct_vacant = B25002_003/total_housing_units) %>%
      select(-starts_with("B2"))
    
    #Occupied house tenure 
    print(paste0('Extracting household tenure variables...'))
    tenure =  get_acs(geography = geography, sumfile = 'acs5',
                      variables = c(sapply(seq(1,3,1), function(v) return(paste("B25003_",str_pad(v,3,pad ="0"),sep="")))),
                      year = year, state = state, county = county, geometry = F) %>% 
      select(-moe) %>%
      spread(key = 'variable', value = 'estimate') %>% 
      mutate(house_pct_ownd_occupied = B25003_002 / B25003_001, 
             house_pct_rent_occupied = B25003_003/B25003_001) %>%
      select(-starts_with("B2"))
    
    #Number of rooms in each housing unit 
    print(paste0('Extracting household room number variables...'))
    room =  get_acs(geography = geography, sumfile = 'acs5',
                    variables = c(sapply(seq(1,10,1), function(v) return(paste("B25017_",str_pad(v,3,pad ="0"),sep="")))),
                    year = year, state = state, county = county, geometry = F) %>% 
      select(-moe) %>%
      spread(key = 'variable', value = 'estimate') %>%
      mutate(denom = B25017_001,
             house_pct_1_room = B25017_002/denom, 
             house_pct_2_room = B25017_003/denom, 
             house_pct_3_room = B25017_004/denom, 
             house_pct_4_room = B25017_005/denom, 
             house_pct_5_room = B25017_006/denom, 
             house_pct_6_room = B25017_007/denom, 
             house_pct_7_room = B25017_008/denom, 
             house_pct_8_room = B25017_009/denom, 
             house_pct_9_plus_room = B25017_010/denom) %>% 
      select(-starts_with("B2"), -denom)
    
    #Year structure was built
    print(paste0('Extracting year household was built...'))
    home_year =  get_acs(geography = geography, sumfile = 'acs5',
                    variables = c(sapply(seq(1,11,1), function(v) return(paste("B25034_",str_pad(v,3,pad ="0"),sep="")))),
                    year = year, state = state, county = county, geometry = F) %>%
      select(-moe) %>% spread(key = 'variable', value='estimate') %>%
      mutate(denom=B25034_001,
             house_yr_pct_2014_plus = B25034_002/denom, 
             house_yr_pct_2010_2013 = B25034_003/denom, 
             house_yr_pct_2000_2009 = B25034_004/denom, 
             house_yr_pct_1990_1999 = B25034_005/denom, 
             house_yr_pct_1980_1989 = B25034_006/denom, 
             house_yr_pct_1970_1979 = B25034_007/denom, 
             house_yr_pct_1960_1969 = B25034_008/denom, 
             house_yr_pct_1950_1959 = B25034_009/denom,
             house_yr_pct_1940_1949 = B25034_010/denom, 
             house_yr_pct_earlier_1939 = B25034_011/denom) %>%
      select(-starts_with("B2"), -denom)
    
    #Type of Heating
    print(paste0('Extracting household heating variables...'))
    heat =  get_acs(geography = geography, sumfile = 'acs5',
                  variables = c(sapply(seq(1,10,1), function(v) return(paste("B25040_",str_pad(v,3,pad ="0"),sep="")))),
                   year = year, state = state, county = county, geometry = F) %>%
    select(-moe) %>%
    spread(key = 'variable', value ='estimate') %>%
    mutate(denom = B25040_001,
           heat_pct_utility_gas = B25040_002/denom,
           heat_pct_bottled_tank_lpgas = B25040_003/denom,
           heat_pct_electricity = B25040_004/denom,
           heat_pct_fueloil_kerosene = B25040_005/denom,
           heat_pct_coal = B25040_006/denom,
           heat_pct_wood = B25040_007/denom,
           heat_pct_solar = B25040_008/denom,
           heat_pct_other = B25040_009/denom,
           heat_pct_no_fuel = B25040_010/denom
            ) %>%
    select(-starts_with('B2'), -denom)

    #Plumbing Facilities: complete, or incomplete
    print(paste0('Extracting household plumbing variables...'))
    plumbing =  get_acs(geography = geography, sumfile = 'acs5',
                        variables = c(sapply(seq(1,3,1), function(v) return(paste("B25047_",str_pad(v,3,pad ="0"),sep="")))),
                        year = year, state = state, county = county, geometry = F) %>%
      select(-moe) %>% 
      spread(key = 'variable', value='estimate') %>%
      mutate(denom = B25047_001, 
             house_pct_complete_plumb = B25047_002/denom, 
             house_pct_incomplete_plumb = B25047_003/denom) %>%
      select(-starts_with("B2"), -denom)
    
    
    #Kitchen facilities: complete or incomplete 
    print(paste0('Extracting household kitchen variables...'))
    kitchen =  get_acs(geography = geography, sumfile = 'acs5',
                       variables = c(sapply(seq(1,3,1), function(v) return(paste("B25051_",str_pad(v,3,pad ="0"),sep="")))),
                       year = year, state = state, county = county, geometry = F) %>%
      select(-moe) %>% 
      spread(key = 'variable', value='estimate') %>%
      mutate(denom = B25051_001, 
             house_pct_complete_kitchen = B25051_002/denom, 
             house_pct_incomplete_kitchen = B25051_003/denom) %>%
      select(-starts_with('B2'), -denom)
    
    
    # value of house (owner-occupied houses)
    print(paste0('Extracting household value ($$) variables...'))
    value = get_acs(geography = geography, sumfile = 'acs5', 
                    variables = c(sapply(seq(1,27,1), function(v) return(paste("B25075_",str_pad(v,3,pad ="0"),sep="")))),
                    year = year, state = state, county = county, geometry = F) %>%
      select(-moe) %>%
      spread(key = 'variable', value='estimate') %>% 
      mutate(denom = B25075_001, 
             house_tot_owned = denom,
             house_val_less_10K = B25075_002/denom, 
             house_val_10K_15K = B25075_003/denom, 
             house_val_15K_20K = B25075_004/denom, 
             house_val_20K_25K = B25075_005/denom,
             house_val_25K_30K = B25075_006/denom,
             house_val_30K_35K = B25075_007/denom,
             house_val_35K_40K = B25075_008/denom,
             house_val_40K_50K = B25075_009/denom, 
             house_val_50K_60K = B25075_010/denom, 
             house_val_60K_70K = B25075_011/denom, 
             house_val_70K_80K = B25075_012/denom, 
             house_val_80K_90K = B25075_013/denom,
             house_val_90K_100K = B25075_014/denom, 
             house_val_100K_125K = B25075_015/denom, 
             house_val_125K_150K = B25075_016/denom, 
             house_val_150K_175K = B25075_017/denom, 
             house_val_175K_200K = B25075_018/denom, 
             house_val_200K_250K = B25075_019/denom, 
             house_val_250K_300K = B25075_020/denom, 
             house_val_300K_400K = B25075_021/denom ,
             house_val_400K_500K = B25075_022/denom,
             house_val_500K_750K = B25075_023/denom, 
             house_val_750K_1M = B25075_024/denom, 
             house_val_1M_1.5M = B25075_025/denom, 
             house_val_1.5M_2M = B25075_026/denom, 
             house_val_more_2M =B25075_027/denom) %>%
      select(-denom, -starts_with("B2"))
    
    # Mortgage status 
    print(paste0('Extracting household mortgage variables...'))
    mort = get_acs(geography = geography, sumfile = 'acs5', 
                   variables = c(sapply(seq(1,8,1), function(v) return(paste("B25081_",str_pad(v,3,pad ="0"),sep="")))),
                   state = state, year = year, county = county, geometry = F) %>%
      select(-moe) %>%
      spread(key = 'variable', value='estimate') %>%
      mutate(denom = B25081_001,
             house_tot_w_mort = B25081_002/denom, 
             house_w_1_mort = B25081_007/denom,
             house_w_2_mort = B25081_004/denom,
             house_w_home_equity_loan = B25081_005/denom, 
             house_w_both_2_mort_and_loan = B25081_006/denom, 
             house_no_mort = B25081_008/denom
      ) %>%
      select(-denom, -starts_with("B2"))
    
    
    
    ########## OUTPUT 
    print(paste0('Generating and exporting output...'))
    output <- income %>% merge(age) %>% merge(educ) %>% merge(work) %>% merge(race) %>% merge(house) %>% merge(occupancy) %>% 
              merge(tenure) %>% merge(room) %>% merge(home_year) %>% merge(heat) %>% merge(plumbing) %>% merge(kitchen) %>% 
              merge(mort) %>% merge(value) 
    
    state_stripped = gsub(" ", "", state)
    dst = paste0(outpath,state_stripped,'.csv')
    write.csv(output,dst)

}

    

states <- state.name


lapply(states[41:50], collect_acs_data, geography = 'block group', 
       year = 2017,
       county = NULL,
       outpath = '')

