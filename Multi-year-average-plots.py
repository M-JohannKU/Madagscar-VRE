# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 18:40:37 2024

@author: Sven
"""
import os
import pandas as pd
import openpyxl
import numpy as np



CSTAB_2018 = np.load(r"C:\Users\Asus\Desktop\Thesis\CSTAB_2018.npy",allow_pickle=True)
CSTAB_2019 = np.load(r"C:\Users\Asus\Desktop\Thesis\CSTAB_2019.npy",allow_pickle=True)
CSTAB_2020 = np.load(r"C:\Users\Asus\Desktop\Thesis\CSTAB_2020.npy",allow_pickle=True)
CSTAB_2021 = np.load(r"C:\Users\Asus\Desktop\Thesis\CSTAB_2021.npy",allow_pickle=True)
CSTAB_2022 = np.load(r"C:\Users\Asus\Desktop\Thesis\CSTAB_2022.npy",allow_pickle=True)
CSTAB_2023 = np.load(r"C:\Users\Asus\Desktop\Thesis\CSTAB_2023.npy",allow_pickle=True)


CFw_2018 = np.load(r'C:/Users/Asus/Desktop/Thesis/CF_wind_2018.npy',allow_pickle=True)
CFw_2019 = np.load(r'C:/Users/Asus/Desktop/Thesis/CF_wind_2019.npy',allow_pickle=True)
CFw_2020 = np.load(r'C:/Users/Asus/Desktop/Thesis/CF_wind_2020.npy',allow_pickle=True)
CFw_2021 = np.load(r'C:/Users/Asus/Desktop/Thesis/CF_wind_2021.npy',allow_pickle=True)
CFw_2022 = np.load(r'C:/Users/Asus/Desktop/Thesis/CF_wind_2022.npy',allow_pickle=True)
CFw_2023 = np.load(r'C:/Users/Asus/Desktop/Thesis/CF_wind_2023.npy',allow_pickle=True)


CFs_2018 = np.load(r'C:/Users/Asus/Desktop/Thesis/Solar_CF_2018.npy',allow_pickle=True)
CFs_2019 = np.load(r'C:/Users/Asus/Desktop/Thesis/Solar_CF_2019.npy',allow_pickle=True)
CFs_2020 = np.load(r'C:/Users/Asus/Desktop/Thesis/Solar_CF_2020.npy',allow_pickle=True)
CFs_2021 = np.load(r'C:/Users/Asus/Desktop/Thesis/Solar_CF_2021.npy',allow_pickle=True)
CFs_2022 = np.load(r'C:/Users/Asus/Desktop/Thesis/Solar_CF_2022.npy',allow_pickle=True)
CFs_2023 = np.load(r'C:/Users/Asus/Desktop/Thesis/Solar_CF_2023.npy',allow_pickle=True)



yerr_2018 = np.load(r'C:/Users/Asus/Desktop/Thesis/yerr_2018.npy',allow_pickle=True)
yerr_2019 = np.load(r'C:/Users/Asus/Desktop/Thesis/yerr_2019.npy',allow_pickle=True)
yerr_2020 = np.load(r'C:/Users/Asus/Desktop/Thesis/yerr_2020.npy',allow_pickle=True)
yerr_2021 = np.load(r'C:/Users/Asus/Desktop/Thesis/yerr_2021.npy',allow_pickle=True)
yerr_2022 = np.load(r'C:/Users/Asus/Desktop/Thesis/yerr_2022.npy',allow_pickle=True)
yerr_2023 = np.load(r'C:/Users/Asus/Desktop/Thesis/yerr_2023.npy',allow_pickle=True)

CSTAB_6_year = np.concatenate((CSTAB_2018, CSTAB_2019,CSTAB_2020,CSTAB_2021,CSTAB_2022,CSTAB_2023))


CFw_6_year = np.concatenate((CFw_2018,CFw_2019,CFw_2020,CFw_2021,CFw_2022,CFw_2023))

CFs_6_year = np.concatenate((CFs_2018,CFs_2019,CFs_2020,CFs_2021,CFs_2022,CFs_2023))


yerr_6_year = np.concatenate((yerr_2018, yerr_2019,yerr_2020,yerr_2021,yerr_2022,yerr_2023))

import matplotlib.pyplot as plt

# Sample data
x = CFw_6_year  # X-axis values
y = CSTAB_6_year  # Y-axis values

# Plotting the scatter plot
plt.scatter(x, y, label='Scatter Plot')
#plt.scatter(x,)

# Adding labels to the axes
plt.xlabel('CFw')
plt.ylabel('Cstab')

# Adding a title to the plot
#plt.title('stability factor vs wind capacity factor')

plt.figure(figsize=(5, 3))

plt.legend()


# Displaying the plot
plt.show()

### six year avergae

all_CFw_arrays = np.array([CFw_2018,CFw_2019,CFw_2020,CFw_2021,CFw_2022,CFw_2023])

avg_CFw = np.nanmean(all_CFw_arrays, axis = 0)

all_CFs_arrays = np.array([CFs_2018,CFs_2019,CFs_2020,CFs_2021,CFs_2022,CFs_2023])

avg_CFs = np.nanmean(all_CFs_arrays, axis = 0)


all_cstab_arrays = np.array([CSTAB_2018, CSTAB_2019,CSTAB_2020,CSTAB_2021,CSTAB_2022,CSTAB_2023])

avg_Cstab = np.nanmean(all_cstab_arrays, axis = 0)

all_yerr_arrays = np.array([yerr_2018, yerr_2019,yerr_2020,yerr_2021,yerr_2022,yerr_2023])

avg_yerr = np.nanmean(all_yerr_arrays, axis=0)

#### Ctab vs months
months = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

Cstab = avg_Cstab

yerr = avg_yerr

plt.figure(figsize=(5, 3))

plt.plot(months, Cstab,color = 'orange', label='Cstab', marker='o')

#plt.ylim(0,1)
plt.errorbar(months,Cstab, yerr, color = 'orange')
plt.xlabel('Months')
plt.ylabel('Stability Factor')
#plt.title('Monthly ERA5 stabilty factors 2018-2023')

plt.grid(True)
plt.legend()
#plt.save('Monthly Capacity Factor')
plt.show()


### CF solar and wind 6 year 


yerrs = np.nanstd(CFs_6_year, axis = 0)
yerrw = np.nanstd(CFw_6_year, axis= 0)

plt.figure(figsize=(5,3))

plt.plot(months, avg_CFs, label='CFs', marker='o',color = 'orange')
plt.plot(months, avg_CFw, label='CFw', marker='s', color = 'blue')

#plt.ylim(0,1)
plt.errorbar(months,avg_CFs, yerrs, color = 'orange')
plt.errorbar(months,avg_CFw,yerrw, color = 'blue')
plt.xlabel('Months')
plt.ylabel('Capacity Factor')
#plt.title('Monthly stabilty factors 2018')

plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
#plt.save('Monthly Capacity Factor')
plt.show()


# point 1 0.2764913536513114, 0.2826639551158653

# point 2 

# target_value = 0.3

# # Define a proximity range
# proximity_range = 0.1

# # Find indices of elements within the proximity range of the target value
# indices = np.where(np.abs(CSTAB_6_year - target_value) <= proximity_range)


# print("Indices of elements within proximity of", target_value, ":", indices)
# print("Values within proximity:", CFw_6_year
# [indices])



# import matplotlib.pyplot as plt


# # Create a scatter plot
# fig, ax = plt.subplots()
# scatter = ax.scatter(x, y)

# # Annotate each point with its coordinates
# for i, txt in enumerate(zip(x, y)):
#     ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha='center')

# # Adding labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Scatter Plot with Point Coordinates')

# # Display the plot
# plt.show()
