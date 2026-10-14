ORNL DAAC Home NASA Home
 About Get Data Submit Data Tools Resources Help Sign
Us in
Search ORNL DAAC  Search
DAAC Home > Get Data > NASA Projects > Daymet > User guide
Daymet: Daily Surface Weather Data on a 1-km Grid for North
America, Version 4 R1
Get Data
Documentation Revision Date: 2023-03-06
Dataset Version: 4.5
Summary
This dataset provides Daymet Version 4 R1 data as gridded estimates of daily weather parameters for North America, Hawaii, and Puerto Rico. Daymet
variables include the following parameters: minimum temperature, maximum temperature, precipitation, shortwave radiation, vapor pressure, snow water
equivalent, and day length. The dataset covers the period from January 1, 1980, to December 31 (or December 30 in leap years) of the most recent full
calendar year for the Continental North America and Hawaii spatial regions. Data for Puerto Rico is available starting in 1950. Each subsequent year is
processed individually at the close of a calendar year. Daymet variables are provided as individual files, by variable and year, at a 1 km x 1 km spatial
resolution and a daily temporal resolution. Areas of Hawaii and Puerto Rico are available as files separate from the continental North America. Data are in
a North America Lambert Conformal Conic projection and are distributed in a standardized Climate and Forecast (CF)-compliant netCDF file format. In
Version 4 R1 (ver 4.4), all 2020 and 2021 files were updated to improve predictions especially in high-latitude areas. It was found that input files used for
deriving 2020 and 2021 data had, for a significant portion of Canadian weather stations, missing daily variable readings for the month of January. NCEI
has corrected issues with the Environment Canada ingest feed which led to the missing readings. The revised 2020 and 2021 Daymet V4 R1 files were
derived with new GHCNd inputs. Files outside of 2020 and 2021 have not changed from the previous V4 release.
Daymet V4 daily gridded data are available for download from the ORNL DAAC through website search and order tools. In addition, the ORNL DAAC
supports a separate Daymet Project website which provides customized tools for accessing the data. The Daymet algorithms and data processing
implemented for V4 are described in detail in the related publication Thornton et al. (2021).
Files are available in netCDF format for three separate spatial areas: continental North America (including Canada, the United States, and Mexico),
Hawaii, and Puerto Rico.
There are a total of 1,113 files in netCDF file format. One file by year and spatial region for each of the seven Daymet parameters.
Figure 1. Annual climatologies of 2019 maximum temperature and precipitation derived from Daymet V4 daily data.
Citation
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022. Daymet: Daily Surface Weather Data on a 1-km Grid for North
America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129

Table of Contents
1. Dataset Overview
2. Data Characteristics
3. Application and Derivation
4. Quality Assessment
5. Data Acquisition, Materials, and Methods
6. Data Access
7. References
8. Dataset Revisions
1. Dataset Overview
This dataset provides Daymet Version 4 R1 data as gridded estimates of daily weather parameters for North America, Hawaii, and Puerto Rico. Daymet
variables include the following parameters: minimum temperature, maximum temperature, precipitation, shortwave radiation, vapor pressure, snow water
equivalent, and day length. The dataset covers the period from January 1, 1980, to December 31 (or December 30 in leap years) of the most recent full
calendar year for the Continental North America and Hawaii spatial regions. Data for Puerto Rico is available starting in 1950. Each subsequent year is
processed individually at the close of a calendar year. Daymet variables are provided as individual files, by variable and year, at a 1 km x 1 km spatial
resolution and a daily temporal resolution. Areas of Hawaii and Puerto Rico are available as files separate from the continental North America. Data are in
a North America Lambert Conformal Conic projection and are distributed in a standardized Climate and Forecast (CF)-compliant netCDF file format. In
Version 4 R1 (ver 4.4), all 2020 and 2021 files were updated to improve predictions especially in high-latitude areas. It was found that input files used for
deriving 2020 and 2021 data had, for a significant portion of Canadian weather stations, missing daily variable readings for the month of January. NCEI
has corrected issues with the Environment Canada ingest feed which led to the missing readings. The revised 2020 and 2021 Daymet V4 R1 files were
derived with new GHCNd inputs. Files outside of 2020 and 2021 have not changed from the previous V4 release.
Daymet V4 daily gridded data are available for download from the ORNL DAAC through website search and order tools. In addition, the ORNL DAAC
supports a separate Daymet Project website which provides customized tools for accessing the data. The V4 data supersede all V3 products. The Daymet
algorithms and data processing implemented for V4 are described in detail in the related publication Thornton et al. (2021).
Files are available in netCDF format for three separate spatial areas: continental North America (including Canada, the United States, and Mexico),
Hawaii, and Puerto Rico.
Project: Daymet
Daymet provides long-term, continuous, gridded estimates of daily weather and climatology variables by interpolating and extrapolating ground-based
observations through statistical modeling techniques. The Daymet data products provide driver data for biogeochemical terrestrial modeling and have
myriad applications in many Earth science, natural resource, biodiversity, and agricultural research areas. Daymet weather variables include daily
minimum and maximum temperature, precipitation, vapor pressure, shortwave radiation, snow water equivalent, and day length produced on a 1 km x 1
km gridded surface over continental North America and Hawaii from 1980 and over Puerto Rico from 1950 through the end of the most recent full calendar
year.
Daymet is a research product of the Environmental Sciences Division at Oak Ridge National Laboratory, Oak Ridge, TN. Daymet is supported
by NASA through the Earth Science Data and Information System (ESDIS) and the Terrestrial Ecology Program. Daymet algorithm and processing
development is also supported by the Office of Biological and Environmental Research within the U.S. Department of Energy's Office of Science.
Related Publication
Thornton, P. E., R. Shrestha, M. Thornton, S. Kao, Y. Wei, and B. E. Wilson. 2021. Gridded daily weather data for North America with comprehensive
uncertainty quantification. Scientific Data 8:190. https://doi.org/10.1038/s41597-021-00973-0
Related Datasets
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022. Daymet: Annual Climate Summaries on a 1-km Grid for North
America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2130
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S. Kao, and B.E. Wilson. 2022. Daymet: Monthly Climate Summaries on a 1-km Grid for North
America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2131
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S. Kao, and B.E. Wilson. 2022. Daymet: Station-Level Inputs and Cross-Validation for North America,
Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2132
Acknowledgments
The continued development of the Daymet algorithm and processing is supported in part by the Energy Exascale Earth System Model (E3SM) project,
funded by the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research.
Puerto Rico data for Daymet V4 is available for an extended temporal period starting in 1950. Support was provided by the DOE NGEE Tropics Phase 2
funding to extend the temporal period.
2. Data Characteristics
Spatial Coverage: Continental North America (Mexico, United States, Canada), Hawaii, and Puerto Rico
Spatial Resolution: 1 km grid
Temporal Coverage
Continental North America (Mexico, United States, Canada) and Hawaii: 1980 to 2022
Puerto Rico: 1950 to 2022
Temporal Resolution: Daily
Study Areas (All latitude and longitude given in decimal degrees)

Site Westernmost Longitude Easternmost Longitude Northernmost Latitude Southernmost Latitude
Continental North America -178.1333 -53.0567 82.9143 14.0749
Puerto Rico -67.9927 -64.1196 19.9382 16.8444
Hawaii -160.3056 -154.772 23.5186 17.9539
Data File Information
Filename format: daymet_v4_daily_<region>_<pppp>_<yyyy>.nc, where
<region> is either 'na' (for continental North America), 'hi' (for hawaii), or 'pr' (for Puerto Rico),
<pppp> is the respective parameter abbreviation (dayl, prcp, srad, swe, tmax, tmin, and vp), and
<yyyy> is year (1950 through 2022).
Example filename: daymet_v4_daily_na_prcp_2014.nc
Table 1. Parameters, abbreviations, units, and descriptions.
Parameter Abbreviation Units Description
Duration of the daylight period in seconds per day. This calculation is based on the period of the
Day length dayl s d-1
day during which the sun is above a hypothetical flat horizon
Daily total precipitation in millimeters. Sum of all forms of precipitation converted to a water-
Precipitation prcp mm
equivalent depth.
Incident shortwave radiation flux density in watts per square meter, taken as an average over
Shortwave
srad W m-2 the daylight period of the day. Note: Daily total radiation (MJ/m2/day) can be calculated as
radiation
follows: ((srad (W/m2) * dayl (s/day)) / l,000,000)
Snow water Snow water equivalent in kilograms per square meter. The amount of water contained within the
swe kg m-2
equivalent snowpack.
Maximum
degrees
air tmax Daily maximum 2 m air temperature in degrees Celsius.
C
temperature
Minimum air degrees
tmin Daily minimum 2 m air temperature in degrees Celsius.
temperature C
Water vapor
vp Pa Water vapor pressure in pascals. Daily average partial pressure of water vapor.
pressure
Coordinate Reference System
Projection System: Lambert Conformal Conic
Parameters
Projection units: meters
Datum (spheroid): WGS_84
1st standard parallel: 25 deg N
2nd standard parallel: 60 deg N
Central meridian: -100 deg (W)
Latitude of origin: 42.5 deg N
False easting: 0
False northing: 0
PROJ.4: +proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs
The Daymet Calendar
The Daymet calendar is based on a standard calendar year. All Daymet years, including leap years, have 1–365 days. For leap years, the Daymet data
include leap day (February 29) and December 31 is discarded from leap years to maintain a 365-day year.
Version Information
The data are stored and distributed as individual CF-compliant netCDF files for each parameter. The most current Daymet data are being delivered to the
user in both Daymet software and Daymet data versions. Version information is recorded in the header file of each netCDF file in the Global Attribute
fields: Version_software and Version_data. All Daymet data are provisional and subject to revision.
3. Application and Derivation
The Daymet data have broad applications over a wide variety of research fields including hydrology, terrestrial vegetation growth models, carbon cycle
science, and regional to large scale climate change analysis. Figure 2 shows an example of a regional view of an extreme precipitation event that took
place in August of 2017. Four consecutive days of daily total precipitation are shown when Hurricane Harvey made landfall in Texas and
Louisiana. Measurements of near-surface meteorological conditions are made at many locations, but researchers are often faced with having to perform
ecosystem process simulations in areas where no meteorological measurements have been taken. The continuous gridded surfaces of the Daymet
dataset were developed to overcome these limitations.

Figure 2. A regional view of Daymet V4 daily total precipitation seen over four days during the time period that Hurricane Harvey hit landfall on Texas and
Louisiana in August of 2017.
4. Quality Assessment
Within the Daymet algorithm, the Daymet cross-validation analyses were used to test the sensitivity of Daymet methods to the variation of parameters and
to estimate the prediction errors associated with the final selected parameters. The general cross-validation protocol is to withhold one observation at a
time from a sample, generate a prediction error for the withheld case by comparison with observed values within the sample, and repeat over all
observations in the sample to generate an average prediction error. Interest is in both the absolute value and the sign of prediction errors generated in
this manner.

5. Data Acquisition, Materials, and Methods
All versions of the Daymet algorithm, including this Version 4, use as its core input daily observations of minimum temperature, maximum temperature,
and precipitation from a network of ground-based weather stations. The surface weather input observations for processing Daymet were obtained from the
NOAA National Centers for Environmental Information's Global Historical Climatology Network (GHCN)-Daily dataset (Menne et al., 2012a). Three
separate input station files were generated for the continental North America, Puerto Rico, and Hawaii study areas and these three spatially distinct areas
were applied separately through the Daymet algorithm.
GHCN-Daily Version Download Dates:
Daymet V4 for years 1980 - 2019 (or 1950 - 2019 (Puerto Rico only)) used GHCN Daily version 3.27 downloaded on February 5, 2020.
Daymet V4 for year 2020 used GHCN Daily version 3.28 downloaded on February 14, 2021.
Daymet V4 for year 2021 used GHCN Daily version 3.28 downloaded on February 16, 2022.
Daymet V4 R1 for year 2020 used GHCN Daily version 3.29 downloaded on October 6, 2022.
Daymet V4 R1 for year 2021 used GHCN Daily version 3.29 downloaded on September 18, 2022
Daymet V4 R1 for year 2022 used GHCN Daily version 3.29 downloaded on February 11, 2023
The Daymet approach to estimating daily surface weather parameters at locations lacking instrumentation is based on a combination of interpolation and
extrapolation, using inputs from multiple instrumented sites and weights for each site that reflect the spatial and temporal relationships of the estimation
location to the instrumental observations. The approximate number of instrumental observations to use for each estimation is defined as a parameter for
each of the primary Daymet variables. As part of a series of algorithm modifications intended to improve robustness in regions of very low station density,
the Daymet V4 algorithm drops the iterative station density calculation and instead defines a search radius for each estimation location which is sized to
capture exactly the average number of input stations, based on pre-calculated arrays of station distances. Given the pre-processed input station
observations and the pre-calculated station lists and interpolation weights for each location in the estimation grid, two separate workflows are used to
produce the primary Daymet output variables: one for the daily temperature variables (Tmax and Tmin) and another for the daily precipitation variable
(Prcp).
The Daymet Version 4 methodology and dataset is described in Thornton, et al. (2021).
In addition to daily maximum and minimum temperature and daily total precipitation, the Daymet data record includes estimates of other important surface
weather quantities that are not routinely observed, or are available as observations from only a small fraction of the temperature and precipitation
observing stations. These secondary output variables are daily total shortwave radiation (Srad), daily average water vapor pressure (VP), duration of the
daylight period (daylength), and a simple estimate of accumulated snowpack, measured as snowpack water equivalent (SWE). The daylength estimate is
based on geographic location and time of year. Estimates for the other secondary variables (Srad, VP, and SWE) are derived from the primary
temperature and precipitation variables on the basis of theory and empirical relationships, as further described in Thornton et al. (2021).
6. Data Access
These data are available through the Oak Ridge National Laboratory (ORNL) Distributed Active Archive Center (DAAC).
Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1
Contact for Data Center Access Information:
E-mail: uso@daac.ornl.gov
Telephone: +1 (865) 241-3952
7. References
Menne, M.J., I. Durre, B. Korzeniewski, S. McNeal, K. Thomas, X. Yin, S. Anthony, R. Ray, R.S. Vose, B.E. Gleason, and T.G. Houston 2012a. Global
Historical Climatology Network - Daily (GHCN-Daily), Version 3. NOAA National Climatic Data Center. Version 3.27-upd-
2020020523. https://doi.org/10.7289/V5D21VHZ
Menne, M.J., I. Durre, R.S. Vose, B.E. Gleason, and T.G. Houston. 2012b. An overview of the Global Historical Climatology Network-Daily
Database. Journal of Atmospheric and Oceanic Technology 29:897-910. http://doi.org/10.7289/V5D21VHZ
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022. Daymet: Annual Climate Summaries on a 1-km Grid for North
America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2130
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S. Kao, and B.E. Wilson. 2022. Daymet: Monthly Climate Summaries on a 1-km Grid for North
America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2131
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S. Kao, and B.E. Wilson. 2022. Daymet: Station-Level Inputs and Cross-Validation for North America,
Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2132
Thornton, P. E., R. Shrestha, M. Thornton, S. Kao, Y. Wei, and B. E. Wilson. 2021. Gridded daily weather data for North America with comprehensive
uncertainty quantification. Scientific Data 8:190. https://doi.org/10.1038/s41597-021-00973-0
8. Dataset Revisions
ORNL DAAC Version Record for Version 4
Daymet ORNL
Product DAAC Description
Version Release
Date
Version March 1,
ORNL DAAC released daily gridded mosaics for 2022 for each Daymet Version 4 variable.
4.5 2023

Daymet ORNL
Product DAAC Description
Version Release
Date
In this Version 4.4, all 2020 and 2021 files (42 total) were updated to improve predictions especially in high-
latitude areas. It was found that input files used for deriving 2020 and 2021 data had, for a significant portion of
Canadian weather stations, missing daily variable readings for the month of January. The missing readings have
Version November been verified by the National Centers for Environmental Information (NCEI) who maintain the Global Historical
4.4 1, 2022 Climatology Network daily (GHCNd) datasbase. NCEI has corrected issues with the Environment Canada ingest
feed which led to the missing readings and has built in more protections to prevent the loss of data. The revised
2020 and 2021 Daymet V4 files were derived with new GHCNd inputs. Files outside of 2020 and 2021 have not
changed.
Version Version 4.3 of the monthly and annual climatologies was an update to cloud optimized GeoTIFF. Because the
4.3 crossval and daily data do not have any GeoTIFF files, there was no version 4.3 issued for these data products.
Update
to March 9,
ORNL DAAC released daily gridded mosaics for 2021 for each Daymet Version 4 variable.
Version 2022
4.2
Update
to April 8,
ORNL DAAC released daily gridded mosaics for 2020 for each Daymet Version 4 variable.
Version 2021
4.1
Version December ORNL DAAC released daily gridded mosaics for Continental North America (Mexico, United States, Canada) and
4.0 15, 2020 Hawaii for 1980–2019 and for Puerto Rico for 1950 - 2019 for each Daymet Version 4 variable.
ORNL DAAC Version Record for Version 3
ORNL
Daymet
DAAC
Product Description
Release
Version
Date
Version March
ORNL DAAC released daily gridded mosaics for 2019 for each Daymet Version 3 variable.
3.4 17, 2020
Version April 3,
ORNL DAAC released daily gridded mosaics for 2018 for each Daymet Version 3 variable.
3.3 2019
Version April 19,
ORNL DAAC released daily gridded mosaics for 2017 for each Daymet Version 3 variable.
3.2 2018
Version April 20,
ORNL DAAC released daily gridded mosaics for 2016 for each Daymet Version 3 variable.
3.1 2017
ORNL DAAC released new daily gridded mosaics for all meteorological variables and all years (1980-2015) based
Version July 15,
on improvements to the Daymet algorithm and expanded geographic coverage at high latitude (see below).
3.0 2016
The Daymet website was also updated. Daymet V2 mosaics are now available only upon request.
ORNL DAAC Version Record for Version 2
ORNL DAAC
Daymet Product
Release Description
Version
Date
March 7,
2015 mosaics ORNL DAAC released daily gridded mosaics for 2015 for each Daymet variable.
2016
2014 mosaics May 18, 2015 ORNL DAAC released daily gridded mosaics for 2014 for each Daymet variable.
June 13,
2013 mosaics ORNL DAAC released daily gridded mosaics for 2013 for each Daymet variable.
2014
Version 2, North
ORNL DAAC archived and released new daily gridded mosaics for meteorological parameters and
American May 15, 2014
updated Daymet Website. All existing Daymet products and tools continue to be available.
mosaics
NACP Modeling and Synthesis Thematic Data Center (MAST-DC) at ORNL released new daily
May 10, 2012
gridded meteorological parameter data for 2-degree Daymet tiles through the Daymet Website.
    
Privacy Policy | Feedback | Help

Home About Us Get Data Submit Data Tools Resources  Contact
| Mission | Science Themes | Submit Data Form | MODIS | Learning | Us  |
| ------- | -------------- | ---------------- | ----- | -------- | --- |
Data Use and Citation NASA Projects Data Scope and THREDDS Data Management
| Policy | All Datasets | Acceptance | SDAT | News |     |
| ------ | ------------ | ---------- | ---- | ---- | --- |
User Working Group Data Authorship Policy Daymet Earthdata Forum 
| Partners |     | Data Publication Timeline | Airborne Data Visualizer |     |     |
| -------- | --- | ------------------------- | ------------------------ | --- | --- |
|          |     | Detailed Submission       | Soil Moisture Visualizer |     |     |
|          |     | Guidelines                | Land - Water Checker     |     |     |