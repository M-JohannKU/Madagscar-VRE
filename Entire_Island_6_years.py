  # -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:04:49 2023

@author: Sven
"""
##### THIS WAS XARRAY heavy version, heavily edited and revised with Sebatian September 10 2023, mostly replace with numpy

##### MASTER copy of thesis work, others are experimental/ideas

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import geopandas as gpd
from scipy.interpolate import make_interp_spline
from scipy.stats import iqr
import cartopy.feature as cfeature
from shapely.geometry import box
import cartopy.feature as feature
import cartopy.io.shapereader as shapereader
import netCDF4
import rioxarray as rio
from shapely.geometry import mapping

##for plotting


# ### Selecting data source  #### use 2018 as it is the year the MSR is based on
# filename = r"C:\Users\Asus\Desktop\Thesis\ERA5\Madagascar_5_years_componants\MAD_2018_5year_componant.nc"

# #designation of datasource
# ds = xr.open_dataset(filename, decode_times=False)


### Selecting data source
#filename = r"C:\Users\Asus\Desktop\Thesis\ERA5\2017\Senegal_2017.nc"

filename_2023 = r"C:\Users\Asus\Desktop\Thesis\ERA5\Madagascar_5_years_componants\MAD_2023_5year_componant.nc"
filename_2022 = r"C:\Users\Asus\Desktop\Thesis\ERA5\Madagascar_5_years_componants\MAD_2022_5year_componant.nc"
filename_2021 = r"C:\Users\Asus\Desktop\Thesis\ERA5\Madagascar_5_years_componants\MAD_2021_5year_componant.nc"
filename_2020 = r"C:\Users\Asus\Desktop\Thesis\ERA5\Madagascar_5_years_componants\MAD_2020_5year_componant.nc" #leap year 2012,2016,2020
filename_2019 = r"C:\Users\Asus\Desktop\Thesis\ERA5\Madagascar_5_years_componants\MAD_2019_5year_componant.nc"
filename_2018 = r"C:\Users\Asus\Desktop\Thesis\ERA5\Madagascar_5_years_componants\MAD_2018_5year_componant.nc"

#filename_HYDRA = r"C:/Users/Asus/Desktop/Thesis/ERA5/MAD_5yr_hydra_error.nc"

#designation of datasource
#ds = xr.open_dataset(filename, decode_times=False)


ds1 = xr.open_dataset(filename_2018, decode_times=False)
ds2 = xr.open_dataset(filename_2019, decode_times=False)
ds3 = xr.open_dataset(filename_2020, decode_times=False)
ds4 = xr.open_dataset(filename_2021, decode_times=False)
ds5 = xr.open_dataset(filename_2022, decode_times=False)
ds6 = xr.open_dataset(filename_2023, decode_times=False)

ds6 = ds6.reduce(np.nansum, dim='expver',keep_attrs=True)
#####concat

#ds = xr.concat([ds1, ds2, ds3, ds4, ds5], dim = 'time') ## COncat produces the proper time bits

ds = xr.concat([ds2, ds3, ds4, ds5,ds6], dim = 'time') ## COncat produces the proper time bits



############# MASKING

prctile_low = 25
prctile_high = 75
 



ds.rio.set_spatial_dims(x_dim="latitude", y_dim="longitude", inplace=True)
ds.rio.write_crs("epsg:4326", inplace=True)
Africa_Shape = gpd.read_file(r"C:\Users\Asus\Desktop\Thesis\Shapefiles\Madagascar\Madagascar.shx", crs="epsg:4326")

clipped = ds.rio.clip(Africa_Shape.geometry.apply(mapping), Africa_Shape.crs, drop=False)


###Extracting Variables UNMASKED



u10 = clipped['u10'] 
u10 = u10.data

v10 = clipped['v10']
v10 = v10.data

u100 = clipped['u100']
u100 = u100.data

v100 = clipped['v100']
v100 = v100.data

t2m = clipped['t2m']
t2m = t2m.data

ssrd = clipped['ssrd']
ssrd = ssrd.data


#### notes for sebastian
##### lat and lon fn.data seems to work, but for individual data, like u10 and ssrd only yields nan values

#wind detials, separated by convention (Sebatians MATLAB)
latitude = clipped['latitude']
#latitude = latitude.data

longitude = clipped['longitude']
#longitude = longitude.data

time = clipped['time']
#time = time.data


##### Leap year accomodation
if time.any() > 1051893:
    leap_year = True
elif time.any() < 1060768:
    leap_year = True
else:
    leap_year = False

    
########################################
#Wind Setup

simulation_years = np.array([2016,2017,2018,2019,2020,2021,2022,2023])
months = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
months_num = np.array(range(1, len(months)+1))
if leap_year == True:
    days = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
# elif filename_2020:
#     np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
else:
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


########################################
#Wind stuff

simulations_years = np.array([2016,2017,2018,2019,2020,2021,2022,2023])
months = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
months_num = np.array(range(1, len(months)+1))
days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
dayhrs_wind = 24
nighthrs = list(range(0, 8)) + list(range(20, 23))
#adjusted nighthrs from 1,2,3,4,5,6,7,8,20,21,22,23,24 to 0,1,2,3,4,5,6,7,8,20,21,22,23 because of indexing
hrs_wind = list(range(dayhrs_wind)) 

positions_wind = np.zeros(len(days)+1)
positions_wind[0] = 1
for n in range(0,len(days)):
    positions_wind[n+1] = dayhrs_wind*days[n] + positions_wind[n]

positions_wind_int = positions_wind.astype(int)

##absolute windspeeds
map_10m_windspeed = (u10**2 + v10**2)**0.5
map_100m_windspeed = (u100**2 + v100**2)**0.5



condition = map_100m_windspeed < map_10m_windspeed
map_100m_windspeed = xr.where(condition, map_10m_windspeed, map_100m_windspeed)




##################################

# Surface roughness somewhere in here
#sebastian says ignore roughness just use 100m for turbine height

##############################

#Solar Radiation

#lots of conversions to get the ssrd the same resolution as wind
#now all data is hourly and spatially consistent from ERA5 
#convert T2m to celsius
tC = t2m - 273.15

GHI = ssrd/(3600)

hours = 24

dayhrs_solar = dayhrs_wind
hrs_solar = hrs_wind
    
#revised v/ copilot
positions_solar = np.zeros(len(days)+1)
positions_solar[0] = 1
for n in range(0,len(days)):
    positions_solar[n+1] = dayhrs_solar*days[n] + positions_solar[n]
    
positions_solar_int = positions_solar.astype(int)


    # constants (Wild et al. 2011)
eta_ref = 1.0;         # definition
GHI_standard = 10**3;   # nominal test conditions for PV panels sept 23, when written as 10^3 returns 9, changed to 10**3 for 1000
beta = 0.0045;         # typical for monocrystalline silicon cells
gamma = 0.1;           # "
T_ref = 25;            # "
c1 = -3.75;            # ", unit: degC
c2 = 1.14;             # ", unitless
c3 = 0.0175;           # ", unit: degC*m^2/W


 #FOR TAKING OUT TEMPERATURE EFFECT
 
#map_t2m = 25 + 0.*map_t2m;

map_GHI = GHI
# remember: in these vectors, 1) month, 2) hour, 3) day
# cell temperature (Wild et al. 2011)

T_cell = c1 + c2*tC + c3*map_GHI

# efficiency of cell (Wild et al. 2011)
eta_cell = eta_ref*(1 - beta*(T_cell - T_ref) + gamma*np.log(map_GHI)/np.log(10))
eta_cell = np.abs(eta_cell)


eta_cell = xr.where(np.isinf(eta_cell), 0, eta_cell)


# # ACTUAL SOLAR POWER OUTPUT
 # as  a percentage this is capacity factor

map_output_solar = (eta_cell*map_GHI)/GHI_standard  ### <- <- this is CFsun!!!!!

map_output_solar = xr.where(map_output_solar > 1, 1, map_output_solar)

# # means of solar output (capacity factor)

map_GHI_mean = np.nanmean(map_GHI, axis = 0)
map_CF_mean_solar = np.nanmean(map_output_solar, axis = 0)



###### WIND POWER

# threshold capacity factor
CF_threshold = 0.0

# 1 = Eol Senegal 500, 2 = Inclin 600; 3 = Yellow Sand;
# 4 = Ecotecnia 62; 5 = Ecotecnia 80; 6 = Repower; 7 = Vestas V126
# 8 = dummy at 10m
#prompt = 'wind turbine type: ';
#x = input(prompt); using python notation not matlab
x = 6

h_0 = 10           # m, where data are from

# wind curve type (see Ould Bilal et al. 2013 & Vestas specs.)
v_in_array = np.array([2,3.5,3,3,3,4,3,3])                  # m/s;
v_rated_array = np.array([9,11,8,13.5,14,13,12,12])            # "
v_out_array = np.array([12,13,15,25,25,25,22.5,22.5])              # "
P_rated_array = np.array([0.5,0.6,0.3,1250,1670,2000,3300,1])   # kW
h_hub_array = np.array([18,7,12,60,70,100,117,10])              # m

# flaw in intial conversion using round brackets for arrays and forgetting ; from matlab
v_in = v_in_array[x]           # pull relevant numbers from array based on turbine selection
v_rated = v_rated_array[x]    # idem
v_out = v_out_array[x]        # idem
P_rated = P_rated_array[x]    # idem
h_hub = h_hub_array[x]        # idem

h_0 = 10  # m, where data are from
h_hub_array = [18, 7, 12, 60, 70, 100, 117, 10]  # m

map_windspeed_10m = map_10m_windspeed
map_windspeed_100m = map_100m_windspeed


map_output_wind = np.zeros(map_windspeed_100m.shape)

                

map_output_wind = xr.where(map_windspeed_100m > v_in, P_rated * ((map_windspeed_100m ** 3 - v_in ** 3) / (v_rated ** 3 - v_in ** 3)), map_output_wind)

map_output_wind = xr.where(map_windspeed_100m > v_rated, P_rated, map_output_wind)

map_output_wind = xr.where(map_windspeed_100m > v_out, 0, map_output_wind)


##CF what is happening here
map_output_wind = map_output_wind / P_rated #### <- <- this is CFWIND !!!!!




map_CF_mean_wind = np.nanmean(map_output_wind, axis = 0)

# Set values less than CF_threshold to NaN
map_CF_mean_wind = xr.where(map_CF_mean_wind < CF_threshold, np.nan, map_CF_mean_wind)

#### Analyze and plot ghi, WINDSPEED, can capacity factors, manual line by line

# #preallocate
# ####Assuming you have 'u10' as a DataArray and 'months' as a list of month values
# # ###Get the dimensions of 'u10' for latitude and longitude
lat_dim = len(latitude)
lon_dim = len(longitude)


####dummies to make xarray of appropriate size to then fill with zeroes
dummy_latlonmon = np.full((u10.shape[1],u10.shape[2],len(months_num)),np.nan)

dummy_latlon = np.full((u10.shape[1], u10.shape[2]),np.nan)

# sun dummies to make xarray of appropriate size to then fill with zeroes
dummy_monhrs_sun = np.full((len(months_num), dayhrs_solar), np.nan)

dummy_mondays = np.full((len(months_num), np.max(days)), np.nan)

av_output_solar_month = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
av_windspeed_month = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))



av_solar_month = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
                    
median_months_output_solar = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
av_windspeed_month = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
av_output_wind_month = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
av_output_wind_month_night = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
#coeff_stab_diurnal = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
coeff_stab_diurnal =  np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
coeff_stab_seasonal = np.zeros(shape = (np.size(dummy_latlon,0), np.size(dummy_latlon,1)))
coeff_syn_diurnal = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
coeff_syn_seasonal = np.zeros(shape = (np.size(dummy_latlon,0), np.size(dummy_latlon,1)))

#coeff_stab_diurnal_monthly = np.full((len(months_num), max(days)), np.nan)
coeff_stab_diurnal_monthly = np.zeros( shape = 12)
test_monthly = np.zeros( shape = 12)
matlab_coeff_stab_diurnal = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))

av_windspeed_all =np.zeros(shape = (np.size(dummy_latlon,0), np.size(dummy_latlon,1)))
hr_max_windspeed = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
av_output_wind_all =np.zeros(shape = (np.size(dummy_latlon,0), np.size(dummy_latlon,1))) 
av_coeff_stab_diurnal_all = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1)))

max_output_wind_all =np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
max_coeff_stab_diurnal_all = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1)))

#numpy method #use this method
bymonth_loc_GHI = np.full((len(months_num), dayhrs_solar, max(days)), np.nan)
bymonth_loc_t2m = np.full((len(months_num), dayhrs_solar, max(days)), np.nan)
bymonth_loc_output_solar = np.full((len(months_num), dayhrs_solar, max(days)), np.nan)

### Sept 23 Try replacing bymonth_loc NaNs with zero arrays like everything else



#### spet 7 2023, replaced dayhrs_solar with hours for error reasons, returne to dayhrs_solar after sept 19




av_GHI_hr = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
#
av_GHI_day = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
upper_GHI = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
lower_GHI =np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))

av_t2m_hr = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
av_t2m_day = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
upper_t2m = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
lower_t2m = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))


# wind dummies to make xarray of appropriate size to then fill with zeroes
###replacing dayhrs_wind with hours
dummy_monhrs_wind = np.full((len(months_num), hours), np.nan)

#bymonth_loc_windspeed_10m = xr.DataArray(np.nan, dims=("months","hours","days"),
                                #      coords={"months": months, "hours": hours,"days": days})
bymonth_loc_winddir_10m = np.full((len(months_num), dayhrs_solar, max(days)), np.nan)
#bymonth_loc_h_rough = xr.DataArray(np.nan, dims=("months","hours","days"),
#                                     coords={"months": months, "hours": dayhrs_wind,"days": days})
bymonth_loc_output_wind = np.full((len(months_num), dayhrs_solar, max(days)), np.nan)

bymonth_loc_windspeed_10m = np.full((len(months_num), hours, max(days)), np.nan)

bymonth_loc_windspeed_100m = np.full((len(months_num), hours, max(days)), np.nan)

av_windspeed_hr = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))

## not in original preallocate had to add
median_windspeed_hr = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
std_months_output_solar = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))

####  old setup in case code breaksstd_months_output_solar = np.full((u10.shape[1],u10.shape[2],len(months_num)),np.nan)
######
av_windspeed_day = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
upper_wind = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
lower_wind = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))

av_output_solar_hr = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
av_output_solar_day = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
upper_output_solar =np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
lower_output_solar = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))

av_output_wind_hr = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
median_output_wind_hr = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))

###### not in original preallocate, added

median_months_output_wind = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))
std_months_output_wind = np.zeros(shape = (np.size(dummy_latlonmon,0), np.size(dummy_latlonmon,1), np.size(dummy_latlonmon,2)))


####
av_output_wind_day =  np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
av_output_wind_day_night =np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
upper_output_wind = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
lower_output_wind = np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
                                   
####stuff DIurnal

variance_solar_diurnal = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
variance_wind_diurnal =  np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
variance_combo_diurnal =  np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
syn_diurnal = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))

av_output_combo_hr =np.zeros(shape = (np.size(dummy_monhrs_sun,0), np.size(dummy_monhrs_sun,1)))
av_output_combo_day = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))                         


# share in capacity mix
cap_share_solar = 1
cap_share_wind = 1

#start the loop
count = 0 

# for y in range (lat_dim):  #### the size of the lattitude being iterated across
#     for z in range (lon_dim): #### the size of the longituse being iterated across both should be u10.size, but it was bugging
#check for mask
test_solar = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
test_wind = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
test_combo = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))
test_stab = np.zeros(shape = (np.size(dummy_mondays,0), np.size(dummy_mondays,1)))


for y in range (lat_dim):  #### the size of the lattitude being iterated across
    for z in range (lon_dim): #### the size of the longituse being iterated across both should be u10.size, but it was bugging
    #check for mask
        if ~np.isnan(np.mean(u10, axis = 0)[y,z]):
            #actual coordinates
        
            loc_lat = latitude[y]
            loc_long = longitude[z]
            
             
            # define GHI time series
            loc_GHI = map_GHI[:,y,z]
            #loc_GHI = loc_GHI.stack()
            
            # define temperature (2m) time series
            loc_t2m = tC[:,y,z]
            #loc_t2m = loc_t2m.stack()
            
            #define solar PV output series
            loc_output_solar = map_output_solar[:,y,z]
            #loc_output_solar = loc_output_solar.stack() 
            
            # arrange data in arrays by month (1), hour (2), day (3)
            
            # array by month
            
        
        # Loop over days within the current month
           
            for m in range(len(months)):  #range(len(months)):
                    temp_GHI = loc_GHI[positions_solar_int[m]:positions_solar_int[m+1]]
                    temp_t2m = loc_t2m[positions_solar_int[m]:positions_solar_int[m+1]]
                    temp_output_solar = loc_output_solar[positions_solar_int[m]:positions_solar_int[m+1]]
               
                #split out each day by month
                    for p in range(days[m]):
                        pos_start = p * dayhrs_solar
                        pos_end = pos_start + dayhrs_solar
                        #if m == 11 and p == 30:   # for the final hour of the downloaded year
                            #bymonth_loc_GHI[m, 0:23, p] = temp_GHI[pos_start:pos_end]
                        # else:   # for all other hours of the year
                        bymonth_loc_GHI[m,: , p] = temp_GHI[pos_start:pos_end]
                        #if m == 11 and p == 30:   # for the final hour of the downloaded year
                            #bymonth_loc_t2m[m, 0:23, p] = temp_t2m[pos_start:pos_end]
                        #else:   # for all other hours of the year
                        bymonth_loc_t2m[m, :, p] = temp_t2m[pos_start:pos_end]
                        #if m == 11 and p == 30:   # for the final hour of the downloaded year
                           # bymonth_loc_output_solar[m, 0:23, p] = temp_output_solar[pos_start:pos_end]
                        #else:   # for all other hours of the year
                        bymonth_loc_output_solar[m, :, p] = temp_output_solar[pos_start:pos_end]
                       # else:
                           # print("Index out of range for month", m ,'and day',p)
         
        
            temp_GHI = None
            temp_output_solar = None
            temp_t2m = None           
            #### Code runs to this point, many runtime warnings for empyty slices but it runs 
            ###### GHI
            #calculate av and std per month of daily cycle
            for m in range(len(months)):
                
                # average and std by minute for all months
                av_GHI_hr[m,:] = np.nanmean(bymonth_loc_GHI[m,:,:],axis=1) #already on axis 2 so we use axis 1
                
                # average GHI by day for all months
                temp = np.nanmean(bymonth_loc_t2m[m,:,:],axis=0);
                av_t2m_day[m,:] = temp
                
                # Calculate percentile values for specified range
                lower_t2m[m,:] = np.percentile(bymonth_loc_t2m[m,:,:],prctile_low,axis = 1)
                upper_t2m[m,:] = np.percentile(bymonth_loc_t2m[m,:,:],prctile_high,axis = 1)
            
            # monthly averages
            av_months_t2m = np.nanmean(av_t2m_day,axis=0)
            std_months_t2m = np.nanstd(av_t2m_day,axis=0)
            
            temp = None
            
            #### INITIALISE FOR WIND %%%%%
            
            # define wind speed time series
            loc_windspeed_100m = map_windspeed_100m[:,y,z]
        
            
            # define wind turbine output series
            loc_output_wind = map_output_wind[:,y,z]
        
            
            # arrange data in arrays by month (1), hour (2), day (3)
          
            # array by month
            for m in range(len(months)):
                temp_windspeed = loc_windspeed_100m[positions_wind_int[m]:positions_wind_int[m+1]]
          
                temp_output_wind = loc_output_wind[positions_wind_int[m]:positions_wind_int[m+1]]
                
                # split out each day by month
                for p in range(days[m]):
                    pos_start = p * dayhrs_solar
                    pos_end = pos_start + dayhrs_solar
                    #if m == 11 and p == 30:   # for the final hour of the downloaded year
                        #bymonth_loc_windspeed_100m[m, 0:23, p] = temp_windspeed[pos_start:pos_end]
                    #else:   # for all other hours of the year
                    bymonth_loc_windspeed_100m[m,:,p] = temp_windspeed[pos_start:pos_end]
                    #if m == 11 and p == 30:   # for the final hour of the downloaded year
                       # bymonth_loc_output_wind[m, 0:23, p] = temp_output_wind[pos_start:pos_end]
                    #else:   # for all other hours of the year
                    bymonth_loc_output_wind[m,:,p] = temp_output_wind[pos_start:pos_end]
                    #bymonth_loc_output_wind[m,:,p] = mean[temp_output_wind[pos_start:pos_end]]
                
            
            temp_windspeed = None
        
            temp_output_wind = None
             
            ####### WIND SPEED #######
            
            # calculate av and std per month of daily cycle
            windspeed_temp = bymonth_loc_windspeed_100m
        
            
            for m in range(len(months)):
                # average, median, and std by minute for all months
                av_windspeed_hr[m,:] = np.nanmean(windspeed_temp[m,:,:],axis=1)
                median_windspeed_hr[m,:] = np.nanmedian(windspeed_temp[m,:,:],axis = 1)
                hr_max_windspeed[y, z, m] = np.max(av_windspeed_hr[m, :])
        
                
                # average wind speed by day for all months
                temp1 = np.nanmean(windspeed_temp[m,:,:],axis=0)
                av_windspeed_day[m,:] = temp1
                
                # Calculate percentile values for specified range
                lower_wind[m,:] = np.percentile(windspeed_temp[m,:,:],prctile_low,axis=1)
                upper_wind[m,:] = np.percentile(windspeed_temp[m,:,:],prctile_high,axis=1)
               
            # monthly averages
            av_windspeed_month[y,z,:] = np.nanmean(av_windspeed_day,axis=1)
            median_windspeed_month = np.nanmedian(av_windspeed_day,axis=1)
            std_windspeed_month = np.nanstd(av_windspeed_day,axis = 1)
            
            temp1 = None
            
            ######### CAPACITY FACTORS ########
            
            
            ###### SOLAR CAPACITY FACTORS 
            #solar panel output with chosen parameters Wild et al, reference earlier in code
            
            for m in range(len(months)):
                #average, median, and std by minute for all mnths
                av_output_solar_hr[m,:] = np.nanmean(bymonth_loc_output_solar[m,:,:],axis = 1)
                
                #average solar by day for all moths
                temp2 = np.nanmean(bymonth_loc_output_solar[m,:,:], axis = 0)
                av_output_solar_day[m,:] = temp2
                
                #calculate percentile in specified range
                lower_output_solar[m,:] = np.percentile(bymonth_loc_output_solar[m,:,:], prctile_low, axis = 1)
                upper_output_solar[m,:] = np.percentile(bymonth_loc_output_solar[m,:,:], prctile_high, axis = 1)
                
            temp2 = None  
                
            #monthly averages
            av_output_solar_month[y,z,:] = np.nanmean(av_output_solar_day,axis = 1)
            median_months_output_solar[y,z,:] = np.nanmedian(av_output_solar_day, axis = 1)
            std_months_output_solar[y,z,:] = np.nanstd(av_output_solar_day, axis = 1)
            
            ######## WIND CAPCITY FACTOR
            
            # WIND POWER OUTPUT WITH CHOSEN TURBINE (x denotes turbine characteristics earlier)
            
            for m in range(len(months)):
                # average, median, and std by minute for all months
                av_output_wind_hr[m,:] = np.nanmean(bymonth_loc_output_wind[m,:,:],axis = 1)
                median_output_wind_hr[m,:] =np.nanmedian(bymonth_loc_output_wind[m,:,:],axis = 1)
                
                #average wind CF by day for all months
                temp3 = np.nanmean(bymonth_loc_output_wind[m,:,:], axis = 0)
                av_output_wind_day[m,:] = temp3
                
                #average wind CF by night only for all months
                temp4 = np.nanmean(bymonth_loc_output_wind[m,nighthrs,:],axis = 0)
                av_output_wind_day_night[m,:] = temp4
                
                # calculate percentile values for specified range
                lower_output_wind[m,:] = np.percentile(bymonth_loc_output_wind[m,:,:], prctile_low, axis = 1)
                upper_output_wind[m,:] = np.percentile(bymonth_loc_output_wind[m,:,:], prctile_high, axis = 1)
            
            temp3 = None    
            temp4 = None
            # Monthly averages sept 23 these all work
            av_output_wind_month[y,z,:] = np.nanmean(av_output_wind_day,axis = 1)
            av_output_wind_month_night[y,z,:] = np.nanmean(av_output_wind_day_night,axis = 1)
            median_months_output_wind[y,z,:] = np.nanmedian(av_output_wind_day, axis = 1)
            std_months_output_wind[y,z,:] = np.nanstd(av_output_wind_day, axis = 1)
            
            
            ####### COMBINING SOLAR AND WIND FOR THE BIG RESULTS
            
            # normalize and add solar + wind
            bymonth_loc_output_combo = (cap_share_solar*bymonth_loc_output_solar + cap_share_wind*bymonth_loc_output_wind) / (cap_share_solar + cap_share_wind)
            
            #### Upper and lower bounds (percentile ranges), normalized by same factors as combine output
            upper_combopower = (cap_share_solar*upper_output_solar + cap_share_wind*upper_output_wind) / (cap_share_solar + cap_share_wind)
            lower_combopower = (cap_share_solar*lower_output_solar + cap_share_wind*lower_output_wind) / (cap_share_solar + cap_share_wind)
            
            for m in range(len(months)):
                #average and std by minute each month
                av_output_combo_hr[m,:] = np.nanmean(bymonth_loc_output_combo[m,:,:], axis = 1)
                
                #### average combined power output by day for all months
                temp5 = np.nanmean(bymonth_loc_output_combo[m,:,:], axis = 0)
                av_output_combo_day[m,:] = temp5
                
                #### Daily variance (average across 0-24 hrs)
                
                for p in range(days[m]):
                    variance_solar_diurnal[m,p] = np.nanstd((bymonth_loc_output_solar[m,:,p])/ (np.nanmean(bymonth_loc_output_solar[m,:,p])))
                    variance_wind_diurnal[m,p] = np.nanstd((bymonth_loc_output_wind[m,:,p])/ (np.nanmean(bymonth_loc_output_wind[m,:,p])))
                    variance_combo_diurnal[m,p] = np.nanstd((bymonth_loc_output_combo[m,:,p])/ (np.nanmean(bymonth_loc_output_combo[m,:,p])))
                    
                    # test_solar[m,p] = np.nanvar(bymonth_loc_output_solar[m,:,p])
                    # test_wind[m,p] = np.nanvar(bymonth_loc_output_wind[m,:,p])
                    # test_combo[m,p] = np.nanvar(bymonth_loc_output_combo[m,:,p])
                    # test_stab[m,p] = 1 - (test_combo[m,p]/test_solar[m,p])
                    
                    coeff_stab_diurnal[m,p] = 1 - (variance_combo_diurnal[m,p]/variance_solar_diurnal[m,p])
                    
                    #Calculation of coeff of synergy
                    diff_solar = bymonth_loc_output_solar[m,:,p] - np.nanmean(bymonth_loc_output_solar[m,:,p])
                    diff_wind = bymonth_loc_output_wind[m,:,p] -  np.nanmean(bymonth_loc_output_wind[m,:,p])
                    #cc_1 = np.nansum(diff_solar*diff_wind)
                    #cc_2 = np.nansum(diff_solar**2)
                    #cc_3 = np.nansum(diff_wind**2)
                    #syn_diurnal[m,p] = 0.5*((1-cc_1)/(math.sqrt(cc_2*cc_3)))
                    
                   
                
                coeff_stab_diurnal_monthly[m] = np.nanmean(coeff_stab_diurnal[m])
                test_monthly[m] = np.nanmean(test_stab[m])
                #reduction of diurnal variability
                #coeff_stab_diurnal[y,z,m] = 1 - ((np.nanmean(variance_combo_diurnal[m,:]))/(np.nanmean(variance_solar_diurnal[m,:])))
                #coeff_syn_diurnal[y,z,m] = np.nanmean(syn_diurnal[m,:])
                matlab_coeff_stab_diurnal[y,z,m] = 1 - ((np.nanmean(variance_combo_diurnal[m,:]))/(np.nanmean(variance_solar_diurnal[m,:])))
            
                
            #### MONTHLY AVERAGES OF COMBINES SOLAR/WIND output
            av_months_output_combo = np.nanmean(av_output_combo_day, axis = 1)
            std_months_output_combo = np.nanstd(av_output_combo_day,axis = 1)
            
            # SAME CALCULATION as ABOVE, but for seasonal-scale variance instead of diurnal scale
            variance_solar_season = np.nanstd(av_output_solar_month[y,z,:]) / np.nanmean(av_output_solar_month[y,z,:])
            test_var = np.nanvar(av_output_solar_month[y,z,:])
            variance_wind_season = np.nanstd(av_output_wind_month[y,z,:]) / np.nanmean(av_output_wind_month[y,z,:])
            variance_combo_season =np.nanstd((av_months_output_combo) / np.nanmean(av_months_output_combo))
            coeff_stab_seasonal[y,z] = 1-(variance_combo_season/variance_solar_season)
            
            #synergy coefficient diurnal scale
            cc1_season = np.nansum(((av_output_solar_month[y,z,:]-np.nanmean(av_output_solar_month[y,z,:])))*(av_output_wind_month[y,z,:] - np.mean(av_output_wind_month[y,z,:])))
            cc2_season = np.nansum((av_output_solar_month[y,z,:] - np.nanmean(av_output_solar_month[y,z,:])**2))
            cc3_season = np.nansum((av_output_wind_month[y,z,:] - np.nanmean(av_output_wind_month[y,z,:]))**2)
            
            
            ##keep an eye this is weir, check after running on train
            #coeff_syn_seasonal[y,z] = 0.5*(1-(cc1_season/math.sqrt(cc2_season*cc3_season)))
            
            temp5 = None
            
            ###### YEARLY AVERAGE VALUES
            av_windspeed_all[y,z] = np.nanmean(av_windspeed_month[y,z,:])
            av_output_wind_all[y,z] = np.nanmean(av_output_wind_month[y,z,:])  
            max_output_wind_all[y,z] = np.nanmax(av_output_wind_month[y,z,:])
            av_coeff_stab_diurnal_all[y,z] = np.nanmean(coeff_stab_diurnal[m,:])
            max_coeff_stab_diurnal_all[y,z] = np.nanmax(coeff_stab_diurnal[m,:])
            
            count = count +1
            percentage = round((count / (len(u10) * len(u10[0]))) * 100)
            
            
        


## Notes for sebastian
## having issues with 815, coeff stab diurnal for indwexing, but no issues with other averages same indices
### having issues with coeff syn seasonal, setting an array, likewise same setup in similar lines unsure of problem


#### plot scatter of CF vs C_stab


av_windspeed_all_land = av_windspeed_all
av_output_wind_all_land = av_output_wind_all
av_coeff_stab_diurnal_all_land = av_coeff_stab_diurnal_all


max_output_wind_all_land = max_output_wind_all
max_coeff_stab_diurnal_all_land = max_coeff_stab_diurnal_all



hour = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
month_names = np.array(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

# Create a single figure with 12 subplots arranged in a 3x4 grid
fig, axs = plt.subplots(4, 3, figsize=(16, 12))

q75_wind = np.zeros(shape = (len(months_num), len(hour)))
q25_wind = np.zeros(shape = (len(months_num), len(hour)))
q75_solar = np.zeros(shape = (len(months_num), len(hour)))
q25_solar = np.zeros(shape = (len(months_num), len(hour)))

 
# Loop through the months to create subplots
for m in range(12):
    row = m // 3  # Calculate the row index
    col = m % 3   # Calculate the column index
    
    q75_wind[m,:], q25_wind[m,:] = np.percentile(bymonth_loc_output_wind[m,:,0:days[m]],[75,25],axis=1)

    q75_solar[m,:], q25_solar[m,:] = np.percentile(bymonth_loc_output_solar[m,:,0:days[m]],[75,25],axis=1)
    
    # Plot the smoothed line with colored lines
    axs[row, col].plot(hour, av_output_wind_hr[m, :], color='blue', label='Wind', linewidth=2)
    axs[row, col].plot(hour, av_output_solar_hr[m,:], color='yellow', label='Solar', linewidth=2)
    
    #plot the IQR Range as shaded area for both wind and solar
    axs[row, col].fill_between(hour, q25_wind[m,:], q75_wind[m,:], alpha=0.3, color='blue', label='Wind IQR Range')
    axs[row, col].fill_between(hour, q25_solar[m,:], q75_solar[m,:], alpha=0.3, color='yellow', label='Solar IQR Range', linestyle='dashed')
    
    
    # Adding labels and title for the current subplot
    axs[row, col].set_xlabel('Hours')
    axs[row, col].set_ylabel('CF')
   # axs[row, col].set_title(f'Month {m+1}')
    axs[row, col].set_title(month_names[m])
    
    # Set axis limits for the current subplot
    axs[row, col].set_xlim(0, 24)
    axs[row, col].set_ylim(0, 1)
    axs[row,col].set_yticks([0,0.5,1])
    
# Add a common legend for all subplots
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)  # Adjust the parameters as needed

fig.suptitle('ERA5 Wind and Solar Daily Capacity Factors 2018-2022 entire island', fontsize=16)

# Adjust spacing between subplots
plt.tight_layout()

#saving figure 
plt.savefig('Madagascar fill_between', dpi=300, bbox_inches='tight')

# Show the figure with all subplots
plt.show()



# Create the scatter plot CF S,W per month
#using point of interest from other areas and entire island avg is wonky and not useful
Solar_CF = av_output_solar_month[31,16,:]
Wind_CF = av_output_wind_month[31,16,:]

plt.figure(figsize=(10, 6))

plt.plot(months, Solar_CF, label='Solar', marker='o', linestyle='-')
plt.plot(months, Wind_CF, label='Wind', marker='o', linestyle='-')

plt.xlabel('Months')
plt.ylabel('Capacity Factor')
plt.title('Daily ERA5 averages of Solar and Wind Capacity 2018-2022')
plt.grid(True)
plt.legend()
#plt.save('Monthly Capacity Factor')
plt.show()

#### Ctab vs months
Cstab = coeff_stab_diurnal_monthly

yerr = np.nanstd(coeff_stab_diurnal, axis = 1)

plt.figure(figsize=(10, 6))

plt.plot(months, Cstab, label='Cstab', marker='o')

#plt.ylim(0,1)
plt.errorbar(months,Cstab, yerr)
plt.xlabel('Months')
plt.ylabel('Stability Factor')
plt.title('Monthly ERA5 Stabilty Factors 2018-2022')

plt.grid(True)
plt.legend()
#plt.save('Monthly Capacity Factor')
plt.show()



# Mock-up data (replace with your actual data loading process)
elric = map_output_wind
lat_atae = np.linspace(-10, -30, 81)
lon_atae = np.linspace(43, 53, 41)
time_atae = np.arange(43824)

data_xr = xr.DataArray(elric, coords={'y': lat_atae, 'x': lon_atae, 'time': time_atae},
                        dims=["time", "y", "x"])

db = data_xr
db_mean = db.mean('time', skipna = True)
db_toplot = db_mean.transpose('y', 'x')  # Transpose to match the expected order of lat, lon in plot function


db_toplot.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
db_toplot.rio.write_crs("epsg:4326", inplace=True)
Africa_Shape = gpd.read_file(r"C:\Users\Asus\Desktop\Thesis\Shapefiles\Madagascar\Madagascar.shx", crs="epsg:4326")

elricplot = db_toplot.rio.clip(Africa_Shape.geometry.apply(mapping), Africa_Shape.crs, drop=False)



#title = 'ERA5 Wind Capacity Factor 2018-2022'
cmap = 'Blues'
cbar_label = 'Capacity Factor'
projection = ccrs.PlateCarree()


fig = plt.figure(figsize=(15, 6))
ax = plt.subplot(111, projection=projection, frameon=False)
ax.set_extent([43, 51, -28, -10], ccrs.PlateCarree())  # Adjusted extent values

im = elricplot.plot(ax=ax, cmap=cmap, extend='both', add_colorbar=False, add_labels=False)
cb = plt.colorbar(im, fraction=0.02, pad=0.04, extend='both')
cb.set_label(label=cbar_label, size=15)
cb.ax.tick_params(labelsize=12)

# ax.set_title(title, loc='center', fontsize=16)

ax.coastlines(color='dimgray', linewidth=0.5)

title_x = 0.5  # X-coordinate for centering title
title_y = 1.05  # Y-coordinate for centering title
#ax.text(title_x, title_y, title, transform=ax.transAxes, fontsize=20, ha='center', va='center')


import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import mapping



# Coordinates of the pixel you want to highlight
highlight_pixel = (47, -17.75)

# Plotting code
fig = plt.figure(figsize=(15, 6))
ax = plt.subplot(111, projection=projection, frameon=False)
ax.set_extent([43, 51, -28, -10], ccrs.PlateCarree())

# Plot the dataset
im = elricplot.plot(ax=ax, cmap=cmap, extend='both', add_colorbar=False, add_labels=False)

# Highlight the single pixel
highlight_lon, highlight_lat = highlight_pixel
ax.scatter(highlight_lon, highlight_lat, color='red', marker='o', s=100, transform=ccrs.PlateCarree())

# Colorbar and other plotting details
cb = plt.colorbar(im, fraction=0.02, pad=0.04, extend='both')
cb.set_label(label=cbar_label, size=15)
cb.ax.tick_params(labelsize=12)

ax.coastlines(color='dimgray', linewidth=0.5)

title_x = 0.5
title_y = 1.05
# ax.text(title_x, title_y, title, transform=ax.transAxes, fontsize=20, ha='center', va='center')

plt.show()

#### My map from scratch

x = av_output_wind_month
tx = np.transpose(x, (2, 1, 0))
tx.shape




##check if MSR wind 99 and MSR sun 117 ohysically overlap in space and are value for comparison
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Point 1
lat1, lon1 = -17.8656, 47.03262
area1 = 21

# Point 2
lat2, lon2 = -17.97, 47.039
area2 = 658

# Define the radii of the circles based on the areas
radius1 = (area1 / 3.14159) ** 0.5  # Assuming circular areas
radius2 = (area2 / 3.14159) ** 0.5

# Create Shapely Point objects
point1 = Point(lon1, lat1).buffer(radius1)
point2 = Point(lon2, lat2).buffer(radius2)

# Check if the two points (circles) intersect
overlap = point1.intersects(point2)

if overlap:
    print("The two points overlap.")
else:
    print("The two points do not overlap.")
    



elessar = map_output_solar
lat_at = np.linspace(-10, -30, 81)
lon_at = np.linspace(43, 53, 41)
time_at = np.arange(43824)

data1_xr = xr.DataArray(elessar, coords={'y': lat_at, 'x': lon_at, 'time': time_at},
                        dims=["time", "y", "x"])

da1 = data1_xr

da1_mean = da1.mean('time', skipna = True)

da1_toplot = da1_mean.transpose('y', 'x')  # Transpose to match the expected order of lat, lon in plot function


da1_toplot.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
da1_toplot.rio.write_crs("epsg:4326", inplace=True)
Africa_Shape = gpd.read_file(r"C:\Users\Asus\Desktop\Thesis\Shapefiles\Madagascar\Madagascar.shx", crs="epsg:4326")


elessarplot = da1_toplot.rio.clip(Africa_Shape.geometry.apply(mapping), Africa_Shape.crs, drop=False)


#title = 'ERA5 Solar Capacity Factor 2018-2022'
cmap = 'Reds'
cbar_label = 'Capacity Factor'
projection = ccrs.PlateCarree()


fig = plt.figure(figsize=(15, 6))
ax = plt.subplot(111, projection=projection, frameon=False)
ax.set_extent([43, 51, -28, -10], ccrs.PlateCarree())  # Adjusted extent values

im = elessarplot.plot(ax=ax, cmap=cmap, extend='both', add_colorbar=False, add_labels=False)
cb = plt.colorbar(im, fraction=0.02, pad=0.04, extend='both')
cb.set_label(label=cbar_label, size=15)
cb.ax.tick_params(labelsize=12)

# ax.set_title(title, loc='center', fontsize=16)

ax.coastlines(color='dimgray', linewidth=0.5)

title_x = 0.5  # X-coordinate for centering title
title_y = 1.05  # Y-coordinate for centering title
#ax.text(title_x, title_y, title, transform=ax.transAxes, fontsize=20, ha='center', va='center')



import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import mapping



# Coordinates of the pixel you want to highlight
highlight_pixel = (47, -17.75)

# Plotting code
fig = plt.figure(figsize=(15, 6))
ax = plt.subplot(111, projection=projection, frameon=False)
ax.set_extent([43, 51, -28, -10], ccrs.PlateCarree())

# Plot the dataset
im = elessarplot.plot(ax=ax, cmap=cmap, extend='both', add_colorbar=False, add_labels=False)

# Highlight the single pixel
highlight_lon, highlight_lat = highlight_pixel
ax.scatter(highlight_lon, highlight_lat, color='blue', marker='o', s=100, transform=ccrs.PlateCarree())

# Colorbar and other plotting details
cb = plt.colorbar(im, fraction=0.02, pad=0.04, extend='both')
cb.set_label(label=cbar_label, size=15)
cb.ax.tick_params(labelsize=12)

ax.coastlines(color='dimgray', linewidth=0.5)

title_x = 0.5
title_y = 1.05
# ax.text(title_x, title_y, title, transform=ax.transAxes, fontsize=20, ha='center', va='center')

plt.show()




# coeff_stab_seasonal[coeff_stab_seasonal<0] = 0

# strider = coeff_stab_seasonal
# lat_stride = np.linspace(-10, -30, 81)
# lon_stride = np.linspace(43, 53, 41)


# dataStride = xr.DataArray(strider, coords={'y': lat_stride, 'x': lon_stride},
#                         dims=[ "y", "x"])

# plotStride = dataStride.transpose('y', 'x')  # Transpose to match the expected order of lat, lon in plot function



# plotStride.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
# plotStride.rio.write_crs("epsg:4326", inplace=True)
# Africa_Shape = gpd.read_file(r"C:\Users\Asus\Desktop\Thesis\Shapefiles\Madagascar\Madagascar.shx", crs="epsg:4326")

# toplotStride = plotStride.rio.clip(Africa_Shape.geometry.apply(mapping), Africa_Shape.crs, drop=False)


# # title = 'ERA5 Stability Factor 2018-2022'
# # cmap = 'Greens'
# # cbar_label = 'Stability Factor'
# # projection = ccrs.PlateCarree()


# # fig = plt.figure(figsize=(15, 6))
# # ax = plt.subplot(111, projection=projection, frameon=False)
# # ax.set_extent([43, 51, -28, -10], ccrs.PlateCarree())  # Adjusted extent values

# # im = toplotStride.plot(ax=ax, cmap=cmap, extend='both', add_colorbar=False, add_labels=False)
# # cb = plt.colorbar(im, fraction=0.02, pad=0.04, extend='both')
# # cb.set_label(label=cbar_label, size=15)
# # cb.ax.tick_params(labelsize=12)

# # # ax.set_title(title, loc='center', fontsize=16)

# # ax.coastlines(color='dimgray', linewidth=0.5)

# # title_x = 0.5  # X-coordinate for centering title
# # title_y = 1.05  # Y-coordinate for centering title
# # ax.text(title_x, title_y, title, transform=ax.transAxes, fontsize=20, ha='center', va='center')



# # plt.show()
