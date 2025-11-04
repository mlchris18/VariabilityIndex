## GOAL - Build specified ARS System
## FILE NAMES CHANGED TO REFLECT 5GW

import assetra
from assetra.system import EnergySystemBuilder
from assetra.units import StochasticUnit
from assetra.units import StorageUnit
from assetra.units import EnergyUnit
from assetra.system import EnergySystem
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np 
from logging import getLogger
import os
import csv
import argparse

from V1_custom_units import CustomStorageUnit, HydroUnit
# Update units and put hydro first 
assetra.units.RESPONSIVE_UNIT_TYPES.insert(2, CustomStorageUnit)
assetra.units.NONRESPONSIVE_UNIT_TYPES.insert(1, HydroUnit)

# Read in arguments from the command prompt
parser = argparse.ArgumentParser(description='Process YEAR, GCM, and Additions.')
parser.add_argument('--year', type=int, required=True, help='Year to be processed')
parser.add_argument('--gcm', type=str, required=True, help='GCM to be processed')
parser.add_argument('--gcm_full', type=str, required=True, help='Full GCM to be processed')
parser.add_argument('--fleet_file', type=str, required=True, help='Fleet file to be processed')
parser.add_argument('--region_name', type=str, required=True, help='Region to be processed')
parser.add_argument('--cap_increase_solar', type=float, required=True, help='Solar increase')
parser.add_argument('--cap_increase_wind', type=float, required=True, help='Wind increase')
parser.add_argument('--solar_dir_name', type=str, help='Directory name for solar capacity increase')
parser.add_argument('--wind_dir_name', type=str, help='Directory name for wind capacity increase')

args = parser.parse_args()

# Debug prints
print(f'Received Year: {args.year}')
print(f'Received GCM: {args.gcm}')
print(f'Received GCM Path: {args.gcm_full}')
print(f'Received Fleet Directory: {args.fleet_file}')
print(f'Received Region Name: {args.region_name}')
print(f'Received Solar Increase: {args.cap_increase_solar}')
print(f'Received Wind Increase: {args.cap_increase_wind}')
print(f'Received Solar Directory Name: {args.solar_dir_name}')
print(f'Received Wind Directory Name: {args.wind_dir_name}')

year = args.year
gcm = args.gcm
gcm_full = args.gcm_full
fleet_file = args.fleet_file
region_name = args.region_name
cap_increase_solar = args.cap_increase_solar
cap_increase_wind = args.cap_increase_wind
solar_dir_name = args.solar_dir_name
wind_dir_name = args.wind_dir_name 

#general generation portfolio read in 
Full_Generation_Portfolio = pd.read_csv(fleet_file)

def getRegional_gens(Full_Generation_Portfolio, region_name): 
    Generation_Portfolio = Full_Generation_Portfolio[Full_Generation_Portfolio['Region'].isin([region_name])]
    solar = Generation_Portfolio.loc[Generation_Portfolio['Technology'].isin(['Solar Photovoltaic', 'Solar Thermal without Energy Storage', 'Solar Thermal with Energy Storage']), ['Summer Capacity (MW)', 'Technology', 'Latitude', 'Longitude', 'Plant Code', 'Nameplate Capacity (MW)', 'Region']]
    wind = Generation_Portfolio.loc[Generation_Portfolio['Technology'] == 'Onshore Wind Turbine', ['Summer Capacity (MW)', 'Technology', 'Latitude', 'Longitude', 'Plant Code', 'Nameplate Capacity (MW)', 'Region']]
    #not including pumped storage right now 
    storage = Generation_Portfolio.loc[
    Generation_Portfolio['Technology'].isin(['Batteries', 'Hydroelectric Pumped Storage']), 
    ['Summer Capacity (MW)', 'Technology', 'Latitude', 'Longitude', 'Plant Code', 'Nameplate Capacity (MW)', 'Region']]
    hydro = Generation_Portfolio.loc[Generation_Portfolio['Technology'] == 'Conventional Hydroelectric', ['Summer Capacity (MW)', 'Generator ID', 'Technology', 'Latitude', 'Longitude', 'Plant Code', 'Nameplate Capacity (MW)', 'Region']]
    # Group by 'ORIS Plant Code' and sum the 'Capacity (MW)'
    hydro = hydro.groupby('Plant Code').agg({
        'Nameplate Capacity (MW)': 'sum', 
        'Summer Capacity (MW)': 'sum',
        'Generator ID': 'first',     # Assuming first GEN ID for rows with the same 'ORIS Plant Code'
        'Region': 'first',
        'Latitude': 'first',
        'Longitude': 'first',
        'Technology':'first'    # Assuming region is the same for rows with the same 'ORIS Plant Code'
    }).reset_index()
    hydro['Plant Code'] = hydro['Plant Code'].astype('Int64')
    thermal = Generation_Portfolio[
        ~Generation_Portfolio['Technology'].isin(['Solar Photovoltaic', 'Solar Thermal without Energy Storage', 'Solar Thermal with Energy Storage',
                                                 'Onshore Wind Turbine','Batteries', 'Hydroelectric Pumped Storage', 'Conventional Hydroelectric'])]
    return Generation_Portfolio, solar, wind, storage, hydro, thermal 

def compute_RH(annual_weather_dataset): 
    e = 0.622 #molecturlar wieght ratio of water to dry air
    #Compute saturation vapor pressure es (T2) using a simplified version of the Goff-Gratch equation
    # Temperature (T2) is assumed to be in Kelvin, and the output pressure (es) is in Pa.
    T2 = annual_weather_dataset['T2'] 
    PSFC = annual_weather_dataset['PSFC']
    Q2 = annual_weather_dataset['Q2']
    es=6.1078 * 100 * np.exp(17.27 * (T2 - 273.15) / (T2 - 35.85))
    qsat=(e * es) / (PSFC - (1 - e) * es)
    annual_weather_dataset['RH']=(Q2 / qsat)
    return annual_weather_dataset

def getWeather_data(gcm_full, year): 
    # load processed power generation dataset (solar cf, wind cf)
    cf_data = xr.open_dataset(f'/nfs/turbo/seas-mtcraig-climate/WRFDownscaled/{gcm_full}/Annual_Solar_Wind/Full_Solar_Wind_CapacityFactors.nc')
    pow_gen_dataset = cf_data.sel(Times = slice(np.datetime64(f'{year}-09-01T00:00:00'), np.datetime64(f'{year+1}-09-01T00:00:00'))).rename({'Times':'time'})
    annual_weather_dataset = xr.open_dataset(f'/nfs/turbo/seas-mtcraig-climate/WRFDownscaled/{gcm_full}/{year}/regrid_{year}_ssp370_d02.nc')
    #realign times
    start_time = pd.Timestamp(f'{year}-09-01 00:00')
    times = pd.date_range(start=start_time, periods=len(annual_weather_dataset.Times), freq='h')
    annual_weather_dataset['Times'] = times
    annual_weather_dataset = annual_weather_dataset.rename({'Times':'time'})
    #computing relative humidty 
    annual_weather_dataset = compute_RH(annual_weather_dataset)
    return annual_weather_dataset, pow_gen_dataset

def get_nearest_hourly_profile(
    latitude: float,
    longitude: float,
    array: xr.DataArray
) -> xr.DataArray:
    """Return time series corresponding to the nearest coordinate in a
    WRF power generation data array.

    Args:
        latitude (float): Latitude relative to equator in degrees
        start_hour (datetime): Longitude relative to meridian in degrees
        array (xr.DataArray): "solar_capacity_factor", "wind_capacity_factor",
            or "temperature" or "relative humidity"

    Returns:
        xr.DataArray: Array with time dimension and datetime coordinates.
    """
    return array.sel(
            lat=latitude, 
            lon=longitude, 
            method="nearest"
        ).squeeze(drop=True)

def get_wrf_power_generation_solar_cf(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, pow_gen_dataset["Solar_CF"])

def get_wrf_power_generation_wind_cf(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, pow_gen_dataset["Wind_CF"])

def get_wrf_power_generation_temperature(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, annual_weather_dataset["T2"])

def get_wrf_power_generation_rh(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, annual_weather_dataset["RH"])

def get_wrf_power_generation_psfc(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, annual_weather_dataset["PSFC"])
import pandas as pd

# load temperature dependent outage rate (tdfor) table
tdfor_table_file = Path("temperature_dependent_outage_rates.csv")
tdfor_table = pd.read_csv(tdfor_table_file, index_col=0)
tdfor_table = tdfor_table / 100 # percentages stored as integers

# create mapping table for tdfor table
tech_categories = {
    "CC" : ['Natural Gas Fired Combined Cycle', ],
    "CT" : ['Natural Gas Fired Combustion Turbine','Landfill Gas'],
    "DS" : ['Municipal Solid Waste','Fossil Waste', 'Natural Gas Internal Combustion Engine', 'Non-Fossil Waste', 'Other Natural Gas'],#"Natural Gas Internal Combustion Engine"],
    "ST" : ['Conventional Steam Coal',  'Natural Gas Steam Turbine', 'Petroleum Liquids'],#"Natural Gas Steam Turbine"],
    "NU" : ["Nuclear", 'Fuel Cell'],
    "HD" : ['Conventional Hydroelectric','Hydroelectric Pumped Storage', 'Biomass','Geothermal', 'Other Waste Biomass', 'Wood/Wood Waste Biomass']
                   # add "Hydroelectric Pumped Storage" in HD on next build out ,
    #"Solar Thermal with Energy Storage","Wood/Wood Waste Biomass"]
}

# create mapping from technology to category
tech_mapping = {tech : cat for cat, techs in tech_categories.items() for tech in techs}

def get_hourly_forced_outage_rate(hourly_temperature: xr.DataArray, technology: str) -> xr.DataArray:
    # index tdfor table by tech
    tdfor_map = tdfor_table[tech_mapping.get(technology, "Other")]
    map_temp_to_for = lambda hourly_temperature: tdfor_map.iloc[
            tdfor_map.index.get_indexer(hourly_temperature, method="nearest")
        ]
    return xr.apply_ufunc(
        map_temp_to_for,
        hourly_temperature
    ).rename("hourly_forced_outage_rate")

def load_gens(unit_count, solar, wind, storage, cap_increase_solar, cap_increase_wind): 
    if cap_increase_solar > 0: 
        #solar  generation load in 
        for _, generator in solar.iterrows():
            generator['Summer Capacity (MW)'] =  generator['Summer Capacity (MW)'] * cap_increase_solar
            # get hourly temperature
            hourly_temperature = get_wrf_power_generation_temperature(
                generator["Latitude"],
                generator["Longitude"]
            )
            # get hourly temperature
            hourly_capacity = get_wrf_power_generation_solar_cf(
                generator["Latitude"],
                generator["Longitude"]
            ) * generator["Summer Capacity (MW)"]

            # map temperature to hourly forced outage rate
            hourly_temperature = hourly_temperature - 273.15 #K to C 
            hourly_forced_outage_rate = get_hourly_forced_outage_rate(hourly_temperature, generator["Technology"])

            # create assetra energy unit
            solar_unit = StochasticUnit(
                    id=unit_count,
                    nameplate_capacity=generator["Summer Capacity (MW)"],
                    hourly_capacity=hourly_capacity.clip(max=generator["Summer Capacity (MW)"]),
                    hourly_forced_outage_rate=hourly_forced_outage_rate
                )
            unit_count += 1

            # add unit to energy system
            builder.add_unit(solar_unit)
        print('Solar Loaded')  
    
    if cap_increase_wind > 0: 
        # add wind
        for _, generator in wind.iterrows():
            generator['Summer Capacity (MW)'] =  generator['Summer Capacity (MW)'] * cap_increase_wind
            # get hourly temperature
            hourly_temperature = get_wrf_power_generation_temperature(
                generator["Latitude"],
                generator["Longitude"]
            )
            # get hourly capacity
            hourly_capacity = get_wrf_power_generation_wind_cf(
                generator["Latitude"],
                generator["Longitude"]
            ) * generator["Summer Capacity (MW)"]

            # map temperature to hourly forced outage rate
            hourly_temperature = hourly_temperature - 273.15 #K to C 
            hourly_forced_outage_rate = get_hourly_forced_outage_rate(hourly_temperature, generator["Technology"])

            # create assetra energy unit
            wind_unit = StochasticUnit(
                    id=unit_count,
                    nameplate_capacity=generator["Summer Capacity (MW)"],
                    hourly_capacity=hourly_capacity,
                    hourly_forced_outage_rate=hourly_forced_outage_rate
                )
            unit_count += 1
            #print(unit_count)
            # add unit to energy system
            builder.add_unit(wind_unit)
        print('Wind Loaded')

        return unit_count 
        
#Initialize System 
builder = EnergySystemBuilder()

# every unit must have a unique id
unit_count = 0


#generation 
Generation_Portfolio, solar, wind, storage, hydro, thermal = getRegional_gens(Full_Generation_Portfolio, region_name) 
print('Made Profiles')

#weather 
annual_weather_dataset, pow_gen_dataset = getWeather_data(gcm_full, year) 
print('Got datasets') 

#load in gens
unit_count = load_gens(unit_count,solar, wind, storage, cap_increase_solar, cap_increase_wind)
print('units loaded') 

# Determine system directory based on provided directory names and print them
if (cap_increase_solar > 0) & (cap_increase_wind > 0): 
    system_dir = Path(f"/nfs/turbo/seas-mtcraig-climate/Martha_Research/ASSETRA_models/ARSTesting/{year}/{gcm}/{region_name}/GW/{gcm}_windsolar{solar_dir_name}_{year}")
    
elif cap_increase_solar > 0:
    system_dir = Path(f"/nfs/turbo/seas-mtcraig-climate/Martha_Research/ASSETRA_models/ARSTesting/{year}/{gcm}/{region_name}/GW/{gcm}_solar{solar_dir_name}_{year}")

elif cap_increase_wind > 0:
    system_dir = Path(f"/nfs/turbo/seas-mtcraig-climate/Martha_Research/ASSETRA_models/ARSTesting/{year}/{gcm}/{region_name}/GW/{gcm}_wind{wind_dir_name}_{year}")

energy_system = builder.build()
energy_system.save(system_dir)