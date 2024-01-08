# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:02:01 2023

@author: Sven
"""

import os
import pandas as pd
import openpyxl
import numpy as np

# Define file paths
# wind_file = 'C:/Users/Asus/Desktop/Thesis/MSR data/Excel files/MSR_WIND.xlsx'
# sun_file = 'C:/Users/Asus/Desktop/Thesis/MSR data/Excel files/SOLAR_PV_MSR.xlsx'
# output_file = 'C:/Users/Asus/Desktop/Thesis/MSR data/Excel files/Capacity_factors.xlsx'

# # Read data from Excel files
# f1 = pd.read_excel(sun_file)
# f2 = pd.read_excel(wind_file)

# df1 = pd.DataFrame(f1)

# df2 = pd.DataFrame(f2)


########Madagscar only files 

solar = r"C:\Users\Asus\Desktop\Thesis\Madagascar-MSR-Solar-CF.xlsx"

wind = r"C:\Users\Asus\Desktop\Thesis\Madagascar-MSR-Wind-CF.xlsx"

f1 = pd.read_excel(solar)

f2 = pd.read_excel(wind)

dfsun = pd.DataFrame(f1)
dfwind = pd.DataFrame(f2)

#pull out a singe row
# Antanarivio row 0 for l MSR 99
#rows/locations may not match up, need to fix/check later
subset_sun = dfsun.iloc[0].values

CF_sun = subset_sun[29:]



subset_wind = dfwind.iloc[0].values

CF_wind = subset_wind[29:]

### extracting all wind cfs

#selected_columns = dfwind.iloc[:, np.r_[2, 3, 29:8789]].values

lon = dfwind.iloc[:,2]
lat = dfwind.iloc[:,3]
time = dfwind.iloc[:,29:]





# Example 2D array
array_2d = time

# Example 1D arrays with the same size as axis 0 of the 2D array
array_1d_a = lat
array_1d_b = lon

array_1d_a_reshaped = array_1d_a.reshape(-1, 1)
array_1d_b_reshaped = array_1d_b.reshape(-1, 1)

# Combine the arrays along axis 0
combined_array = np.vstack([array_1d_a, array_1d_b, array_2d])

# Alternatively, you can use np.concatenate
#combined_array = np.concatenate([array_1d_a.reshape(1, -1), array_1d_b.reshape(1, -1), array_2d], axis=0)

print("Combined Array:")
print(combined_array)


#For MSR matchups
#MSR SUN ID:117. closest ot wind max average CF  MSR wind ID:99
#MSR 

MSR_wind_99 = dfwind.iloc[0].values
MSR_wind_99_array = MSR_wind_99[29:]

MSR_sun_117 = dfsun.iloc[69].values
MSR_sun_117_array = MSR_sun_117[29:]


np.save('CF_sun_MSR.npy', CF_sun)
np.save('CF_wind_MSR.npy', CF_wind)
np.save('CF_sun_all.npy', f1)
np.save('CF_wind_all.npy', f2)
np.save('MSR_wind_99.npy',MSR_wind_99_array)
np.save('MSR_sun_117.npy',MSR_sun_117_array)

array_2d = time

# Example pandas Series
series_a = lat
series_b = lon

# Convert Series to DataFrames
df_a = pd.DataFrame(series_a) # Transpose to have a shape of (1, n)
df_b = pd.DataFrame(series_b)

#
combined_df = pd.concat([df_a, df_b,pd.DataFrame(array_2d)], axis=1)

print("Combined DataFrame:")
print(combined_df)


### alll MSR wind CFs 2018 in time, lat lon format
CFw_all_MSR = combined_df.values


value_to_fill = 0.214377838



# Example array
previous_array = CFw_all_MSR

value_to_fill = 0.214377838

# Determine the axis along which to fill the value
axis_to_fill = 0  # Change this to 1 if you want to fill along the second axis

# Create a new array with the same shape as the previous array
CFsun_all = np.full_like(previous_array, fill_value=value_to_fill)

# Fill the specified axis with the value
CFsun_all[axis_to_fill, :] = value_to_fill


# Convert Series to DataFrames
df_a = pd.DataFrame(lat).T  # Transpose to have a shape of (1, n)
df_b = pd.DataFrame(lon).T

# Convert array_2d to a DataFrame
df_2d = pd.DataFrame(time)

# Check shapes for compatibility
if df_a.shape[0] == df_b.shape[0] == df_2d.shape[0]:
    # Combine the arrays along axis 0 to create a 3D array
    combined_3d_array = np.stack([df_a.values, df_b.values, df_2d.values], axis=0)
    print("Combined 3D Array:")
    print(combined_3d_array)
else:
    print("Shapes are not compatible for stacking.")