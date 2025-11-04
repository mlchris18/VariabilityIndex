###Edited on 9/26 to include 4 hour long battery assumption and change output path 
#### NEW - edited to do all including joint battery 8/5/25 
###Edited to do all expect joint battery 1666MW on 6/19/25 at 1023 
## EDITED PRINT PATH
#necessary units to run code 
import argparse
import assetra
from assetra.system import EnergySystem
from assetra.units import StorageUnit
from pathlib import Path
from assetra.simulation import ProbabilisticSimulation
from assetra.metrics import LossOfLoadHours
import xarray as xr
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Build_ARSStorageUnit import ARSStorageUnit
from V1_custom_units import CustomStorageUnit, HydroUnit, SolarandWindUnit
# changing assetra loaded in units -- add back end storage type, remove normal storage
#adding pumped hydro and hydro units 
assetra.units.RESPONSIVE_UNIT_TYPES.insert(1, ARSStorageUnit)
assetra.units.RESPONSIVE_UNIT_TYPES.pop(0) #removing normaal storage unit 
assetra.units.NONRESPONSIVE_UNIT_TYPES.insert(1, SolarandWindUnit)
assetra.units.NONRESPONSIVE_UNIT_TYPES.insert(2, HydroUnit)

print(assetra.units.RESPONSIVE_UNIT_TYPES)

#read in arguments from command prompt 
parser = argparse.ArgumentParser(description='Process YEAR, GCM, and Additions.')
parser.add_argument('--year', type=int, required=True, help='Year to be processed')
parser.add_argument('--region_name', type=str, required=True, help='Region to be processed')

args = parser.parse_args()

#declaring year and region 
year = args.year
region = args.region_name

#load in energy systems and adding battery functions 
def load_energy_system(path): 
    system_dir = Path(f'{path}')

    if system_dir.exists():
        energy_system = EnergySystem()
        energy_system.load(system_dir)
    else:
        print("Saved system not found. Please create and save this system following the instructions found in the appendix (:")
    print("# of Units:", energy_system.size)
    print("Sys. Capacity (MW):", round(energy_system.system_capacity))
    
    return energy_system

def add_battery_power(net_cap, battery_increase, existing):
    global region
    fleet_file = '2023_EIA860_WECC_generation.csv'
    region_name = region 
    Full_Generation_Portfolio = pd.read_csv(fleet_file)
    Generation_Portfolio = Full_Generation_Portfolio[Full_Generation_Portfolio['Region'].isin([region_name])]
    storage = Generation_Portfolio.loc[
        Generation_Portfolio['Technology'].isin(['Batteries', 'Hydroelectric Pumped Storage']), 
        ['Summer Capacity (MW)', 'Technology', 'Latitude', 'Longitude', 'Plant Code', 'Nameplate Capacity (MW)', 'Region']
    ]
    existing_battery = storage['Summer Capacity (MW)'].sum()
    
    nc = make_battery(net_cap, existing_battery, battery_increase)
    
    return nc

def make_battery(net_hourly_matrix, existing_battery, battery_increase):
    
    trials = net_hourly_matrix.coords['trial']
    results = []

    added_storage_cap = existing_battery + existing_battery * battery_increase #add portion of existing supply 
        
    charge_rate = added_storage_cap
    discharge_rate = added_storage_cap
    charge_capacity = (added_storage_cap*4) #(assuming a four hour battery) 
    roundtrip_efficiency = 0.85

    for trial_id in trials:
        # Select net_hourly for the current trial
        net_hourly = net_hourly_matrix.sel(trial=trial_id)

        unit = ARSStorageUnit(
            id=1,
            nameplate_capacity=added_storage_cap,
            charge_rate=charge_rate,
            discharge_rate=discharge_rate,
            charge_capacity=charge_capacity,
            roundtrip_efficiency=roundtrip_efficiency,
            net_hourly_capacity=net_hourly,
        )

        hourly_capacity = unit._get_hourly_capacity(
            charge_rate,
            discharge_rate,
            charge_capacity,
            roundtrip_efficiency,
            net_hourly,
        )

        # Collect results
        results.append(hourly_capacity)

    # Combine results back into an xarray.DataArray
    result_da = xr.concat(results, dim='trial')
    result_da = result_da.assign_coords(trial=trials)
    
    #print(result_da)  # Printing the final result
    return result_da

def add_systems(gcm):
    global region
    global year
    
    # Construct the directory path based on the inputs
    directory_path = f'/nfs/turbo/seas-mtcraig-climate/Martha_Research/ASSETRA_models/ARSTesting/{year}/{gcm}/{region}/GW'
    print(f"Directory path: {directory_path}")
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return []
    
    try:
        # List all non-hidden directories in the directory
        system_dirs = [
            d for d in os.listdir(directory_path) 
            if os.path.isdir(os.path.join(directory_path, d)) and not d.startswith('.')
        ]

        print(f"Directories found: {system_dirs}")
        
        # Identify and separate the directory that begins with the region name
        primary_dir = next((d for d in system_dirs if d.startswith(region)), None)
        if primary_dir:
            system_dirs.remove(primary_dir)
            print(f"Primary directory: {primary_dir}")

        # Sort the remaining directories alphabetically
        system_dirs = sorted(system_dirs)
        print(f"Directories to load (after primary): {system_dirs}")

    except Exception as e:
        print(f"Error reading directory {directory_path}: {e}")
        return []
    
    loaded_systems = []
    
    # Load the primary directory first if it exists
    if primary_dir:
        try:
            dir_path = os.path.join(directory_path, primary_dir)
            print(f"Loading primary system: {dir_path}")
            system_data = load_energy_system(dir_path)  # Adjust function to handle directories if needed
            loaded_systems.append(system_data)
        except Exception as e:
            print(f"Error loading primary system from directory {dir_path}: {e}")
    
    # Load the remaining directories
    for dir_name in system_dirs:
        try:
            dir_path = os.path.join(directory_path, dir_name)
            print(f"Loading system: {dir_path}")
            system_data = load_energy_system(dir_path)  # Adjust function to handle directories if needed
            loaded_systems.append(system_data)
        except Exception as e:
            print(f"Error loading system from directory {dir_path}: {e}")

    print(f"Total systems loaded: {len(loaded_systems)}")
    return loaded_systems

# actually loading in the systems 

base_tai_system, ars_tai_solar10000MW, ars_tai_solar1000MW, ars_tai_solar1666MW, ars_tai_solar2500MW, ars_tai_solar3333MW, ars_tai_solar333MW, ars_tai_solar5000MW,  ars_tai_solar500MW, ars_tai_wind10000MW,  ars_tai_wind1000MW, ars_tai_wind1666MW, ars_tai_wind2500MW, ars_tai_wind3333MW, ars_tai_wind333MW, ars_tai_wind5000MW, ars_tai_wind500MW,  ars_tai_windsolar1666MW, ars_tai_windsolar2500MW, ars_tai_windsolar3333MW, ars_tai_windsolar333MW, ars_tai_windsolar5000MW, ars_tai_windsolar500MW = add_systems('TAI')

base_mpi_system, ars_mpi_solar10000MW, ars_mpi_solar1000MW, ars_mpi_solar1666MW, ars_mpi_solar2500MW, ars_mpi_solar3333MW, ars_mpi_solar333MW, ars_mpi_solar5000MW, ars_mpi_solar500MW, ars_mpi_wind10000MW, ars_mpi_wind1000MW, ars_mpi_wind1666MW, ars_mpi_wind2500MW, ars_mpi_wind3333MW, ars_mpi_wind333MW, ars_mpi_wind5000MW, ars_mpi_wind500MW, ars_mpi_windsolar1666MW, ars_mpi_windsolar2500MW, ars_mpi_windsolar3333MW, ars_mpi_windsolar333MW, ars_mpi_windsolar5000MW, ars_mpi_windsolar500MW = add_systems('MPI')

base_ec3_system, ars_ec3_solar10000MW, ars_ec3_solar1000MW, ars_ec3_solar1666MW, ars_ec3_solar2500MW, ars_ec3_solar3333MW, ars_ec3_solar333MW, ars_ec3_solar5000MW, ars_ec3_solar500MW, ars_ec3_wind10000MW, ars_ec3_wind1000MW, ars_ec3_wind1666MW, ars_ec3_wind2500MW, ars_ec3_wind3333MW, ars_ec3_wind333MW, ars_ec3_wind5000MW, ars_ec3_wind500MW, ars_ec3_windsolar1666MW, ars_ec3_windsolar2500MW, ars_ec3_windsolar3333MW, ars_ec3_windsolar333MW, ars_ec3_windsolar5000MW, ars_ec3_windsolar500MW = add_systems('EC3')

base_ec3veg_system, ars_ec3veg_solar10000MW, ars_ec3veg_solar1000MW, ars_ec3veg_solar1666MW, ars_ec3veg_solar2500MW, ars_ec3veg_solar3333MW, ars_ec3veg_solar333MW, ars_ec3veg_solar5000MW, ars_ec3veg_solar500MW, ars_ec3veg_wind10000MW, ars_ec3veg_wind1000MW, ars_ec3veg_wind1666MW, ars_ec3veg_wind2500MW, ars_ec3veg_wind3333MW, ars_ec3veg_wind333MW, ars_ec3veg_wind5000MW, ars_ec3veg_wind500MW, ars_ec3veg_windsolar1666MW, ars_ec3veg_windsolar2500MW, ars_ec3veg_windsolar3333MW, ars_ec3veg_windsolar333MW, ars_ec3veg_windsolar5000MW, ars_ec3veg_windsolar500MW = add_systems('EC3veg')

base_miroc6_system, ars_miroc6_solar10000MW, ars_miroc6_solar1000MW, ars_miroc6_solar1666MW, ars_miroc6_solar2500MW, ars_miroc6_solar3333MW, ars_miroc6_solar333MW, ars_miroc6_solar5000MW, ars_miroc6_solar500MW, ars_miroc6_wind10000MW, ars_miroc6_wind1000MW, ars_miroc6_wind1666MW, ars_miroc6_wind2500MW, ars_miroc6_wind3333MW, ars_miroc6_wind333MW, ars_miroc6_wind5000MW, ars_miroc6_wind500MW, ars_miroc6_windsolar1666MW, ars_miroc6_windsolar2500MW, ars_miroc6_windsolar3333MW, ars_miroc6_windsolar333MW, ars_miroc6_windsolar5000MW, ars_miroc6_windsolar500MW = add_systems('MIROC6')


#define simulation and functions 
simulation = ProbabilisticSimulation(
    start_hour=f"{year}-09-01 00:00:00",
    end_hour=f"{year +1}-08-30 23:00:00",
    trial_size=10
)

def BASE_RunSims(system, num): 
    global year 
    global region 
    simulation.assign_energy_system(system[0])
    simulation.run()
    net_hourly_capacity_matrix = simulation.net_hourly_capacity_matrix
    #svaing hourly capacity matrix -- these will nto have battery included
    simulation._hourly_capacity_matrix['unit_type'] = simulation._hourly_capacity_matrix['unit_type'].astype(str) #comvert to string so I can save 
    simulation._hourly_capacity_matrix.to_netcdf(f'/nfs/turbo/seas-mtcraig-climate/Martha_Research/ARS_Results/9.26_Tests/{year}/{region}/{system[1]}_nethourlycap_run{num}.nc')
    #adding battery 
    with_battery_netcap =  net_hourly_capacity_matrix + add_battery_power(net_hourly_capacity_matrix, 0, existing=True)
    with_battery_netcap.to_netcdf(f'/nfs/turbo/seas-mtcraig-climate/Martha_Research/ARS_Results/9.26_Tests/{year}/{region}/{system[1]}_netcap_run{num}.nc')
    # convert net hourly capacity matrix to pandas dataframe with risk hours only
    shortfall_matrix_pd = with_battery_netcap.where(lambda c: c < 0).to_pandas().T.dropna(how="all")
    # get loss of load probability
    loss_of_load_prob = shortfall_matrix_pd.count(axis=1) / shortfall_matrix_pd.shape[1]
    # show top 10 risk hours
    loss_of_load_prob.sort_values(ascending=False)[:10]
    #get lolh 
    LOLH = loss_of_load_prob.sum()
    return with_battery_netcap, net_hourly_capacity_matrix, LOLH

def EXPERIMENT_RunSims(base_sims, system, num):
    global year
    global regi
    #run sims on the resource addition portfolio 
    simulation.assign_energy_system(system[0])
    simulation.run()
    additional_net_hourly_capacity_matrix = simulation.net_hourly_capacity_matrix
    #add to base sims
    total_net_capacity = base_sims + additional_net_hourly_capacity_matrix
   # print(total_net_capacity)
    #add battery 
    total_with_battery_netcap = total_net_capacity + add_battery_power(total_net_capacity, 0, existing=True)
   # print(total_with_battery_netcap)
    total_with_battery_netcap.to_netcdf(f'/nfs/turbo/seas-mtcraig-climate/Martha_Research/ARS_Results/9.26_Tests/{year}/{region}/{system[1]}_netcap_run{num}.nc')
    # convert net hourly capacity matrix to pandas dataframe with risk hours only
    shortfall_matrix_pd = total_with_battery_netcap.where(lambda c: c < 0).to_pandas().T.dropna(how="all")
    # get loss of load probability
    loss_of_load_prob = shortfall_matrix_pd.count(axis=1) / shortfall_matrix_pd.shape[1]
    # show top 10 risk hours
    loss_of_load_prob.sort_values(ascending=False)[:10]
    #get lolh 
    LOLH = loss_of_load_prob.sum()
    return LOLH #, additional_net_hourly_capacity_matrix

def EXPERIMENT_AddBattery(base_sim, battery_increase, num): 
    global year
    global region
    #run battery disbatch 
    battery_netcap = add_battery_power(base_sim, battery_increase[0], existing=False)
    #battery_netcap.to_netcdf(f'/nfs/turbo/seas-mtcraig-climate/Martha_Research/ARS_Results/{year}/{region}/{battery_increase[1]}_netcap_run{num}.nc') -- saving here to know this was the problem bc it was printing just the battery as teh resulst not teh fulll system capacity 
    total_net_capacity = base_sim + battery_netcap
    total_net_capacity.to_netcdf(f'/nfs/turbo/seas-mtcraig-climate/Martha_Research/ARS_Results/9.26_Tests/{year}/{region}/{battery_increase[1]}_netcap_run{num}.nc')
    # convert net hourly capacity matrix to pandas dataframe with risk hours only
    shortfall_matrix_pd = total_net_capacity.where(lambda c: c < 0).to_pandas().T.dropna(how="all")
    # get loss of load probability
    loss_of_load_prob = shortfall_matrix_pd.count(axis=1) / shortfall_matrix_pd.shape[1]
    # show top 10 risk hours
    loss_of_load_prob.sort_values(ascending=False)[:10]
    #get lolh 
    LOLH = loss_of_load_prob.sum()
    return LOLH #, additional_net_hourly_capacity_matrix

def EXPERIMENT_RunJointBatSims(base_sims, system, num): #system here has 3 parts, the arc system for wind/solar, the battery increase and the name
    global year
    global region 
    #run sims on the resource addition portfolio 
    simulation.assign_energy_system(system[0])
    simulation.run()
    additional_net_hourly_capacity_matrix = simulation.net_hourly_capacity_matrix
    #add to base sims
    total_net_capacity = base_sims + additional_net_hourly_capacity_matrix
   # print(total_net_capacity)
    #add battery 
    total_with_battery_netcap = total_net_capacity + add_battery_power(total_net_capacity, system[1], existing=False)
   # print(total_with_battery_netcap)
    total_with_battery_netcap.to_netcdf(f'/nfs/turbo/seas-mtcraig-climate/Martha_Research/ARS_Results/9.26_Tests/{year}/{region}/{system[2]}_netcap_run{num}.nc')
    # convert net hourly capacity matrix to pandas dataframe with risk hours only
    shortfall_matrix_pd = total_with_battery_netcap.where(lambda c: c < 0).to_pandas().T.dropna(how="all")
    # get loss of load probability
    loss_of_load_prob = shortfall_matrix_pd.count(axis=1) / shortfall_matrix_pd.shape[1]
    # show top 10 risk hours
    loss_of_load_prob.sort_values(ascending=False)[:10]
    #get lolh 
    LOLH = loss_of_load_prob.sum()
    return LOLH #, additional_net_hourly_capacity_matrix

def run_simulations(num_runs, base_system, ars_solar, ars_wind, ars_battery, ars_windsolar, ars_joint_battery): #ars_solar, ars_wind, ars_battery, ars_windsolar,
    solar_results = []
    wind_results = []
    battery_results = []
    windsolar_results = []
    joint_battery_results = []
    
    for num in range(num_runs):
        base_sim_bat, base_sim_nobat, base_LOLH = BASE_RunSims(base_system, num)  # Re-run base simulation
        #print(base_sim)
        solar_investment = [base_LOLH]
        wind_investment = [base_LOLH]
        battery_investment = [base_LOLH]
        windsolar_investment = [base_LOLH] 
        joint_battery_investment = [base_LOLH]
       # print(base_LOLH) 
        
        for ars in ars_solar:
            solar_investment.append(EXPERIMENT_RunSims(base_sim_nobat, ars, num))
            #print(v)
            
        for ars in ars_wind:
            wind_investment.append(EXPERIMENT_RunSims(base_sim_nobat, ars, num))
        
        for ars in ars_battery:
            battery_investment.append(EXPERIMENT_AddBattery(base_sim_nobat, ars, num)) # has to be base sim no bat or we double the amount of battery in the system because existing adds with addition batteyr in the function 
        for ars in ars_windsolar: 
            windsolar_investment.append(EXPERIMENT_RunSims(base_sim_nobat, ars, num))
        
        for ars in ars_joint_battery: 
            joint_battery_investment.append(EXPERIMENT_RunJointBatSims(base_sim_nobat, ars, num))
            
        solar_results.append(solar_investment)
        wind_results.append(wind_investment)
        battery_results.append(battery_investment)
        windsolar_results.append(windsolar_investment)
        joint_battery_results.append(joint_battery_investment) 
    
    return np.array(solar_results), np.array(wind_results), np.array(battery_results), np.array(windsolar_results), np.array(joint_battery_results) 

# plotting code would go here, but not yet 

#RUNS
# Parameters for simulation
num_runs = 10  # Number of runs for averaging
#bat percentages in order 333MW, 500MW, 1000MW, 1666MW, 2500MW, 3333MW, 5000MW, 10000MW 
if region =='CAMX': 
    battery_percentages = [0.02812346, 0.04218519, 0.08437039, 0.14061731, 0.21092596, 0.28123462, 0.42185193, 0.84370386]
if region == 'DSW': 
    battery_percentages = [0.24219526, 0.36329289, 0.7265877, 1.21097629, 1.81646443, 2.42195258, 3.63292887, 7.26585773]
if region == 'NWPP': 
    battery_percentages = [0.19287891, 0.28931837, 0.57863673, 0.96439455, 1.44659183, 1.92878911, 2.89318366, 5.78636732]

# GCM 1: Assuming we have ars_solar_X and ars_wind_X for GCM 1
base_tai = (base_tai_system, 'base_tai') 
ars_tai_solar = [(ars_tai_solar333MW, 'tai_solar333MW'), (ars_tai_solar500MW, 'tai_solar500MW'),  (ars_tai_solar1000MW, 'tai_solar1000MW'), (ars_tai_solar1666MW, 'tai_solar1666MW'), (ars_tai_solar2500MW, 'tai_solar2500MW'), (ars_tai_solar3333MW, 'tai_solar3333MW'), (ars_tai_solar5000MW, 'tai_solar5000MW'), (ars_tai_solar10000MW, 'tai_solar10000MW')]
ars_tai_wind = [(ars_tai_wind333MW, 'tai_wind333MW'), (ars_tai_wind500MW, 'tai_wind500MW'), (ars_tai_wind1000MW, 'tai_wind1000MW'), (ars_tai_wind1666MW, 'tai_wind1666MW'), (ars_tai_wind2500MW, 'tai_wind2500MW'), (ars_tai_wind3333MW, 'tai_wind3333MW'), (ars_tai_wind5000MW, 'tai_wind5000MW'), (ars_tai_wind10000MW, 'tai_wind10000MW')]
ars_tai_battery = [(battery_percentages[0], 'tai_battery333MW'), (battery_percentages[1], 'tai_battery500MW'), (battery_percentages[2], 'tai_battery1000MW'), (battery_percentages[3], 'tai_battery1666MW'), (battery_percentages[4], 'tai_battery2500MW'), (battery_percentages[5], 'tai_battery3333MW'), (battery_percentages[6], 'tai_battery5000MW'), (battery_percentages[7], 'tai_battery10000MW')]
ars_tai_windsolar = [(ars_tai_windsolar333MW, 'tai_windsolar333MW'), (ars_tai_windsolar500MW, 'tai_windsolar500MW'), (ars_tai_windsolar1666MW, 'tai_windsolar1666MW'), (ars_tai_windsolar2500MW, 'tai_windsolar2500MW'), (ars_tai_windsolar3333MW, 'tai_windsolar3333MW'), (ars_tai_windsolar5000MW,'tai_windsolar5000MW')]
#ars_tai_joint_battery = [(ars_tai_solar1666MW, battery_percentages[3], 'tai_solarbattery1666MW'), (ars_tai_wind1666MW, battery_percentages[3], #'tai_windbattery1666MW')]
ars_tai_joint_battery = [(ars_tai_solar500MW, battery_percentages[1], 'tai_solarbattery500MW'), (ars_tai_solar2500MW, battery_percentages[4], 'tai_solarbattery2500MW'), (ars_tai_solar5000MW, battery_percentages[6], 'tai_solarbattery5000MW'), (ars_tai_wind500MW, battery_percentages[1], 'tai_windbattery500MW'), (ars_tai_wind2500MW, battery_percentages[4], 'tai_windbattery2500MW'), (ars_tai_wind5000MW, battery_percentages[6], 'tai_windbattery5000MW'),(ars_tai_windsolar333MW, battery_percentages[0], 'tai_windsolarbattery333MW'), (ars_tai_windsolar1666MW, battery_percentages[3], 'tai_windsolarbattery1666MW'), (ars_tai_windsolar3333MW, battery_percentages[5], 'tai_windsolarbattery3333MW'), (ars_tai_solar1666MW, battery_percentages[3], 'tai_solarbattery1666MW'), (ars_tai_wind1666MW, battery_percentages[3], 'tai_windbattery1666MW')]

# GCM 2: Assuming we have ars_solar_X and ars_wind_X for GCM 2
base_mpi = (base_mpi_system, 'base_mpi') 
ars_mpi_solar = [(ars_mpi_solar333MW, 'mpi_solar333MW'), (ars_mpi_solar500MW, 'mpi_solar500MW'),  (ars_mpi_solar1000MW, 'mpi_solar1000MW'), (ars_mpi_solar1666MW, 'mpi_solar1666MW'), (ars_mpi_solar2500MW, 'mpi_solar2500MW'), (ars_mpi_solar3333MW, 'mpi_solar3333MW'), (ars_mpi_solar5000MW, 'mpi_solar5000MW'), (ars_mpi_solar10000MW, 'mpi_solar10000MW')]
ars_mpi_wind = [(ars_mpi_wind333MW, 'mpi_wind333MW'), (ars_mpi_wind500MW, 'mpi_wind500MW'), (ars_mpi_wind1000MW, 'mpi_wind1000MW'), (ars_mpi_wind1666MW, 'mpi_wind1666MW'), (ars_mpi_wind2500MW, 'mpi_wind2500MW'), (ars_mpi_wind3333MW, 'mpi_wind3333MW'), (ars_mpi_wind5000MW, 'mpi_wind5000MW'), (ars_mpi_wind10000MW, 'mpi_wind10000MW')]
ars_mpi_battery = [(battery_percentages[0], 'mpi_battery333MW'), (battery_percentages[1], 'mpi_battery500MW'), (battery_percentages[2], 'mpi_battery1000MW'), (battery_percentages[3], 'mpi_battery1666MW'), (battery_percentages[4], 'mpi_battery2500MW'), (battery_percentages[5], 'mpi_battery3333MW'), (battery_percentages[6], 'mpi_battery5000MW'), (battery_percentages[7], 'mpi_battery10000MW')]
ars_mpi_windsolar = [(ars_mpi_windsolar333MW, 'mpi_windsolar333MW'), (ars_mpi_windsolar500MW, 'mpi_windsolar500MW'), (ars_mpi_windsolar1666MW, 'mpi_windsolar1666MW'), (ars_mpi_windsolar2500MW, 'mpi_windsolar2500MW'), (ars_mpi_windsolar3333MW, 'mpi_windsolar3333MW'), (ars_mpi_windsolar5000MW,'mpi_windsolar5000MW')]

#ars_mpi_joint_battery = [(ars_mpi_solar1666MW, battery_percentages[3], 'mpi_solarbattery1666MW'), (ars_mpi_wind1666MW, battery_percentages[3], #'mpi_windbattery1666MW')]

ars_mpi_joint_battery= [(ars_mpi_solar500MW, battery_percentages[1], 'mpi_solarbattery500MW'), (ars_mpi_solar2500MW, battery_percentages[4], 'mpi_solarbattery2500MW'), (ars_mpi_solar5000MW, battery_percentages[6], 'mpi_solarbattery5000MW'), (ars_mpi_wind500MW, battery_percentages[1], 'mpi_windbattery500MW'), (ars_mpi_wind2500MW, battery_percentages[4], 'mpi_windbattery2500MW'), (ars_mpi_wind5000MW, battery_percentages[6], 'mpi_windbattery5000MW'),(ars_mpi_windsolar333MW, battery_percentages[0], 'mpi_windsolarbattery333MW'), (ars_mpi_windsolar1666MW, battery_percentages[3], 'mpi_windsolarbattery1666MW'), (ars_mpi_windsolar3333MW, battery_percentages[5], 'mpi_windsolarbattery3333MW'), (ars_mpi_solar1666MW, battery_percentages[3], 'mpi_solarbattery1666MW'), (ars_mpi_wind1666MW, battery_percentages[3], 'mpi_windbattery1666MW')]

#GCM 3
base_ec3 = (base_ec3_system, 'base_ec3') 
ars_ec3_solar = [(ars_ec3_solar333MW, 'ec3_solar333MW'), (ars_ec3_solar500MW, 'ec3_solar500MW'),  (ars_ec3_solar1000MW, 'ec3_solar1000MW'), (ars_ec3_solar1666MW, 'ec3_solar1666MW'), (ars_ec3_solar2500MW, 'ec3_solar2500MW'), (ars_ec3_solar3333MW, 'ec3_solar3333MW'), (ars_ec3_solar5000MW, 'ec3_solar5000MW'), (ars_ec3_solar10000MW, 'ec3_solar10000MW')]
ars_ec3_wind = [(ars_ec3_wind333MW, 'ec3_wind333MW'), (ars_ec3_wind500MW, 'ec3_wind500MW'), (ars_ec3_wind1000MW, 'ec3_wind1000MW'), (ars_ec3_wind1666MW, 'ec3_wind1666MW'), (ars_ec3_wind2500MW, 'ec3_wind2500MW'), (ars_ec3_wind3333MW, 'ec3_wind3333MW'), (ars_ec3_wind5000MW, 'ec3_wind5000MW'), (ars_ec3_wind10000MW, 'ec3_wind10000MW')]
ars_ec3_battery = [(battery_percentages[0], 'ec3_battery333MW'), (battery_percentages[1], 'ec3_battery500MW'), (battery_percentages[2], 'ec3_battery1000MW'), (battery_percentages[3], 'ec3_battery1666MW'), (battery_percentages[4], 'ec3_battery2500MW'), (battery_percentages[5], 'ec3_battery3333MW'), (battery_percentages[6], 'ec3_battery5000MW'), (battery_percentages[7], 'ec3_battery10000MW')]
ars_ec3_windsolar = [(ars_ec3_windsolar333MW, 'ec3_windsolar333MW'), (ars_ec3_windsolar500MW, 'ec3_windsolar500MW'), (ars_ec3_windsolar1666MW, 'ec3_windsolar1666MW'), (ars_ec3_windsolar2500MW, 'ec3_windsolar2500MW'), (ars_ec3_windsolar3333MW, 'ec3_windsolar3333MW'), (ars_ec3_windsolar5000MW,'ec3_windsolar5000MW')]

#ars_ec3_joint_battery = [(ars_ec3_solar1666MW, battery_percentages[3], 'ec3_solarbattery1666MW'), (ars_ec3_wind1666MW, battery_percentages[3], #'ec3_windbattery1666MW')]

ars_ec3_joint_battery= [(ars_ec3_solar500MW, battery_percentages[1], 'ec3_solarbattery500MW'), (ars_ec3_solar2500MW, battery_percentages[4], 'ec3_solarbattery2500MW'), (ars_ec3_solar5000MW, battery_percentages[6], 'ec3_solarbattery5000MW'), (ars_ec3_wind500MW, battery_percentages[1], 'ec3_windbattery500MW'), (ars_ec3_wind2500MW, battery_percentages[4], 'ec3_windbattery2500MW'), (ars_ec3_wind5000MW, battery_percentages[6], 'ec3_windbattery5000MW'),(ars_ec3_windsolar333MW, battery_percentages[0], 'ec3_windsolarbattery333MW'), (ars_ec3_windsolar1666MW, battery_percentages[3], 'ec3_windsolarbattery1666MW'), (ars_ec3_windsolar3333MW, battery_percentages[5], 'ec3_windsolarbattery3333MW'), (ars_ec3_solar1666MW, battery_percentages[3], 'ec3_solarbattery1666MW'), (ars_ec3_wind1666MW, battery_percentages[3], 'ec3_windbattery1666MW')]



#GCM 4
base_ec3veg = (base_ec3veg_system, 'base_ec3veg') 

ars_ec3veg_solar = [(ars_ec3veg_solar333MW, 'ec3veg_solar333MW'), (ars_ec3veg_solar500MW, 'ec3veg_solar500MW'),  (ars_ec3veg_solar1000MW, 'ec3veg_solar1000MW'), (ars_ec3veg_solar1666MW, 'ec3veg_solar1666MW'), (ars_ec3veg_solar2500MW, 'ec3veg_solar2500MW'), (ars_ec3veg_solar3333MW, 'ec3veg_solar3333MW'), (ars_ec3veg_solar5000MW, 'ec3veg_solar5000MW'), (ars_ec3veg_solar10000MW, 'ec3veg_solar10000MW')]
ars_ec3veg_wind = [(ars_ec3veg_wind333MW, 'ec3veg_wind333MW'), (ars_ec3veg_wind500MW, 'ec3veg_wind500MW'), (ars_ec3veg_wind1000MW, 'ec3veg_wind1000MW'), (ars_ec3veg_wind1666MW, 'ec3veg_wind1666MW'), (ars_ec3veg_wind2500MW, 'ec3veg_wind2500MW'), (ars_ec3veg_wind3333MW, 'ec3veg_wind3333MW'), (ars_ec3veg_wind5000MW, 'ec3veg_wind5000MW'), (ars_ec3veg_wind10000MW, 'ec3veg_wind10000MW')]
ars_ec3veg_battery = [(battery_percentages[0], 'ec3veg_battery333MW'), (battery_percentages[1], 'ec3veg_battery500MW'), (battery_percentages[2], 'ec3veg_battery1000MW'), (battery_percentages[3], 'ec3veg_battery1666MW'), (battery_percentages[4], 'ec3veg_battery2500MW'), (battery_percentages[5], 'ec3veg_battery3333MW'), (battery_percentages[6], 'ec3veg_battery5000MW'),(battery_percentages[7], 'ec3veg_battery10000MW')]
ars_ec3veg_windsolar = [(ars_ec3veg_windsolar333MW, 'ec3veg_windsolar333MW'), (ars_ec3veg_windsolar500MW, 'ec3veg_windsolar500MW'), (ars_ec3veg_windsolar1666MW, 'ec3veg_windsolar1666MW'), (ars_ec3veg_windsolar2500MW, 'ec3veg_windsolar2500MW'), (ars_ec3veg_windsolar3333MW, 'ec3veg_windsolar3333MW'), (ars_ec3veg_windsolar5000MW,'ec3veg_windsolar5000MW')]

#ars_ec3veg_joint_battery = [(ars_ec3veg_solar1666MW, battery_percentages[3], 'ec3veg_solarbattery1666MW'), (ars_ec3veg_wind1666MW, #battery_percentages[3], 'ec3veg_windbattery1666MW')]


ars_ec3veg_joint_battery = [(ars_ec3veg_solar500MW, battery_percentages[1], 'ec3veg_solarbattery500MW'), (ars_ec3veg_solar2500MW, battery_percentages[4], 'ec3veg_solarbattery2500MW'), (ars_ec3veg_solar5000MW, battery_percentages[6], 'ec3veg_solarbattery5000MW'), (ars_ec3veg_wind500MW, battery_percentages[1], 'ec3veg_windbattery500MW'), (ars_ec3veg_wind2500MW, battery_percentages[4], 'ec3veg_windbattery2500MW'), (ars_ec3veg_wind5000MW, battery_percentages[6], 'ec3veg_windbattery5000MW'),(ars_ec3veg_windsolar333MW, battery_percentages[0], 'ec3veg_windsolarbattery333MW'), (ars_ec3veg_windsolar1666MW, battery_percentages[3], 'ec3veg_windsolarbattery1666MW'), (ars_ec3veg_windsolar3333MW, battery_percentages[5], 'ec3veg_windsolarbattery3333MW'), (ars_ec3veg_solar1666MW, battery_percentages[3], 'ec3veg_solarbattery1666MW'), (ars_ec3veg_wind1666MW, battery_percentages[3], 'ec3veg_windbattery1666MW')]


#GCM 5
base_miroc6 = (base_miroc6_system, 'base_miroc6')

ars_miroc6_solar = [(ars_miroc6_solar333MW, 'miroc6_solar333MW'), (ars_miroc6_solar500MW, 'miroc6_solar500MW'),  (ars_miroc6_solar1000MW, 'miroc6_solar1000MW'), (ars_miroc6_solar1666MW, 'miroc6_solar1666MW'), (ars_miroc6_solar2500MW, 'miroc6_solar2500MW'), (ars_miroc6_solar3333MW, 'miroc6_solar3333MW'), (ars_miroc6_solar5000MW, 'miroc6_solar5000MW'), (ars_miroc6_solar10000MW, 'miroc6_solar10000MW')]
ars_miroc6_wind = [(ars_miroc6_wind333MW, 'miroc6_wind333MW'), (ars_miroc6_wind500MW, 'miroc6_wind500MW'), (ars_miroc6_wind1000MW, 'miroc6_wind1000MW'), (ars_miroc6_wind1666MW, 'miroc6_wind1666MW'), (ars_miroc6_wind2500MW, 'miroc6_wind2500MW'), (ars_miroc6_wind3333MW, 'miroc6_wind3333MW'), (ars_miroc6_wind5000MW, 'miroc6_wind5000MW'), (ars_miroc6_wind10000MW, 'miroc6_wind10000MW')]
ars_miroc6_battery = [(battery_percentages[0], 'miroc6_battery333MW'), (battery_percentages[1], 'miroc6_battery500MW'), (battery_percentages[2], 'miroc6_battery1000MW'), (battery_percentages[3], 'miroc6_battery1666MW'), (battery_percentages[4], 'miroc6_battery2500MW'), (battery_percentages[5], 'miroc6_battery3333MW'), (battery_percentages[6], 'miroc6_battery5000MW'), (battery_percentages[7], 'miroc6_battery10000MW')]
ars_miroc6_windsolar = [(ars_miroc6_windsolar333MW, 'miroc6_windsolar333MW'), (ars_miroc6_windsolar500MW, 'miroc6_windsolar500MW'), (ars_miroc6_windsolar1666MW, 'miroc6_windsolar1666MW'), (ars_miroc6_windsolar2500MW, 'miroc6_windsolar2500MW'), (ars_miroc6_windsolar3333MW, 'miroc6_windsolar3333MW'), (ars_miroc6_windsolar5000MW,'miroc6_windsolar5000MW')]

#ars_miroc6_joint_battery = [(ars_miroc6_solar1666MW, battery_percentages[3], 'miroc6_solarbattery1666MW'), (ars_miroc6_wind1666MW, #battery_percentages[3], 'miroc6_windbattery1666MW')]


ars_miroc6_joint_battery= [(ars_miroc6_solar500MW, battery_percentages[1], 'miroc6_solarbattery500MW'), (ars_miroc6_solar2500MW, battery_percentages[4], 'miroc6_solarbattery2500MW'), (ars_miroc6_solar5000MW, battery_percentages[6], 'miroc6_solarbattery5000MW'), (ars_miroc6_wind500MW, battery_percentages[1], 'miroc6_windbattery500MW'), (ars_miroc6_wind2500MW, battery_percentages[4], 'miroc6_windbattery2500MW'), (ars_miroc6_wind5000MW, battery_percentages[6], 'miroc6_windbattery5000MW'),(ars_miroc6_windsolar333MW, battery_percentages[0], 'miroc6_windsolarbattery333MW'), (ars_miroc6_windsolar1666MW, battery_percentages[3], 'miroc6_windsolarbattery1666MW'), (ars_miroc6_windsolar3333MW, battery_percentages[5], 'miroc6_windsolarbattery3333MW'), (ars_miroc6_solar1666MW, battery_percentages[3], 'miroc6_solarbattery1666MW'), (ars_miroc6_wind1666MW, battery_percentages[3], 'miroc6_windbattery1666MW')]

'''
jointbattery_tai_results = run_simulations(num_runs, base_tai, ars_tai_joint_battery)
print ('TAI Run') 
jointbattery_mpi_results = run_simulations(num_runs, base_mpi, ars_mpi_joint_battery)
print ('MPI Run') 
jointbattery_ec3_results = run_simulations(num_runs, base_ec3, ars_ec3_joint_battery)
print ('EC3 Run') 
jointbattery_ec3veg_results = run_simulations(num_runs, base_ec3veg, ars_ec3veg_joint_battery)
print ('EC3VEG Run') 
jointbattery_miroc6_results = run_simulations(num_runs, base_miroc6, ars_miroc6_joint_battery)
print ('MIROC6 Run') 
'''

# Run the simulations for all GCMs
solar_tai_results, wind_tai_results, battery_tai_results, windsolar_tai_results, jointbattery_tai_results = run_simulations(num_runs, base_tai, ars_tai_solar, ars_tai_wind, ars_tai_battery, ars_tai_windsolar, ars_tai_joint_battery)
print ('TAI Run') 
solar_mpi_results, wind_mpi_results, battery_mpi_results, windsolar_mpi_results, jointbattery_mpi_results= run_simulations(num_runs, base_mpi, ars_mpi_solar, ars_mpi_wind, ars_mpi_battery, ars_mpi_windsolar, ars_mpi_joint_battery)
print ('MPI Run') 
solar_ec3_results, wind_ec3_results, battery_ec3_results, windsolar_ec3_results, jointbattery_ec3_results = run_simulations(num_runs, base_ec3, ars_ec3_solar, ars_ec3_wind, ars_ec3_battery, ars_ec3_windsolar, ars_ec3_joint_battery)
print ('EC3 Run') 
solar_ec3veg_results, wind_ec3veg_results, battery_ec3veg_results, windsolar_ec3veg_results, jointbattery_ec3veg_results = run_simulations(num_runs, base_ec3veg, ars_ec3veg_solar, ars_ec3veg_wind, ars_ec3veg_battery, ars_ec3veg_windsolar, ars_ec3veg_joint_battery)
print ('EC3 veg Run') 
solar_miroc6_results, wind_miroc6_results, battery_miroc6_results, windsolar_miroc6_results, jointbattery_miroc6_results = run_simulations(num_runs, base_miroc6, ars_miroc6_solar, ars_miroc6_wind, ars_miroc6_battery, ars_miroc6_windsolar, ars_miroc6_joint_battery)
print ('MIROC6 Run') 

# Collect all results for plotting
results = [
    (solar_tai_results, wind_tai_results, battery_tai_results, windsolar_tai_results, jointbattery_tai_results, 'TAI'),
    (solar_mpi_results, wind_mpi_results, battery_mpi_results,  windsolar_mpi_results, jointbattery_mpi_results, 'MPI'),
    (solar_ec3_results, wind_ec3_results, battery_ec3_results,  windsolar_ec3_results, jointbattery_ec3_results, 'EC3'),
    (solar_ec3veg_results, wind_ec3veg_results, battery_ec3veg_results, windsolar_ec3veg_results, jointbattery_ec3veg_results, 'EC3VEG'),
    (solar_miroc6_results, wind_miroc6_results, battery_miroc6_results, windsolar_miroc6_results, jointbattery_miroc6_results, 'MIROC6'),
]

import csv
# Determine the maximum length of result sets
max_length = max(len(solar_tai_results), len(wind_tai_results), len(battery_tai_results))

# Open a CSV file to write
with open(f'/nfs/turbo/seas-mtcraig-climate/Martha_Research/ARS_Results/9.26_Tests/{year}/{region}/results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write a header if necessary
    header = ['Solar', 'Wind', 'Battery', 'WindSolar', 'JointBattery', 'Label']
    writer.writerow(header)

    # Write data row by row
    for solar_results, wind_results, battery_results, windsolar_results, jointbattery_results, label in results:
        for i in range(max_length):
            solar_value = solar_results[i] if i < len(solar_results) else ''
            wind_value = wind_results[i] if i < len(wind_results) else ''
            battery_value = battery_results[i] if i < len(battery_results) else ''
            windsolar_value = windsolar_results[i] if i < len(windsolar_results) else ''
            jointbattery_value = jointbattery_results[i] if i < len(jointbattery_results) else ''
            writer.writerow([solar_value, wind_value, battery_value, windsolar_value, jointbattery_value, label])





