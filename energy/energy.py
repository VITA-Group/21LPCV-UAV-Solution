# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:26:36 2021

@author: Administrator
"""

import csv
import numpy as np
import pandas as pd

#Change this line to your original data store path
data_filename = "I:\\ipcv\\powerData\\data.csv"
df = pd.read_csv(data_filename, sep=",")
timestamp = df[['timestamp']].copy()
power_measurements = df[['power']].copy()

#Change this line to the timestamp file
time_stamp_filename = "I:\\ipcv\\powerData\\timestamp.csv"

start_timestamp = []
end_timestamp = []
rows = []
energy_measurement = []

with open(time_stamp_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    _ = next(csvreader)
    for row in csvreader:
        if(len(row)==0):
            continue
        start_timestamp.append(row[0])
        end_timestamp.append(row[1])

for i in range(0, len(start_timestamp)):
    start_time = float(start_timestamp[i])
    end_time = float(end_timestamp[i])
    start_idx = int((timestamp.iloc[(np.searchsorted(df.timestamp.values, start_time) - 1).clip(0)] - 1).name)
    end_idx = int((timestamp.iloc[(np.searchsorted(df.timestamp.values, end_time) - 1).clip(0)]).name)
    print(start_idx, end_idx)
    energy = 0
    energy_sequence = []
    average_scale = []
    time = []
    for idx in range(start_idx, end_idx):
        """
        if timestamp['timestamp'][idx+1] != timestamp['timestamp'][idx]:
            if len(average_scale) == 0:
                energy_sequence.append(power_measurements['power'][idx])
                time.append(timestamp['timestamp'][idx])
            else:
                average_scale.append(power_measurements['power'][idx])
                energy_sequence.append(np.mean(average_scale))
                average_scale = []
                time.append(timestamp['timestamp'][idx])
        else:
            average_scale.append(power_measurements['power'][idx])
        """
        time_interval = timestamp['timestamp'][idx+1] - timestamp['timestamp'][idx]
        energy = energy + (power_measurements['power'][idx] * time_interval)
    """
    if len(average_scale) != 0:
        time.append(timestamp['timestamp'][idx])
        energy_sequence.append(np.mean(average_scale))
    #print(time, energy_sequence)
    for i in range(len(energy_sequence)):
        energy += energy_sequence[i]
    """
    print("time is: ",str(timestamp['timestamp'][end_idx] - timestamp['timestamp'][start_idx]))
    print("Energy: " + str(energy))
    print("Energy Density: " + str(energy/(timestamp['timestamp'][end_idx] - timestamp['timestamp'][start_idx])))
    energy_measurement.append(energy)

print("Energy Consumption: " + str(sum(energy_measurement)))
print("Number of Measurements: " + str(len(energy_measurement)))
#print("Average Energy Consumption: " + str(sum(energy_measurement) / len(energy_measurement)))


