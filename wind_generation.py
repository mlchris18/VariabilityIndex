import numpy as np
import xarray as xr

def calc_wind_power(wind_power_input, powercurve=None, windhgt=100,
                    windspeed_name='was', windspeed100m='100mWind',
                    tas='tas', ps='ps', huss='huss'):
    """Calculate wind power for a location based time series of 100-m wind speed data
    Follows same procedure as https://www.nature.com/articles/s41561-017-0029-9#Sec5
    The calculated quantity is the power output from a single turbine of this type.

    In:

    Out:

    """

    # Convert inputs to required variable names
    surface_temp = wind_power_input[tas]
    # Need temperature in Kelvin to apply gas law
    if surface_temp.attrs['units'] == 'deg C':
        surface_temp = surface_temp + 273.15

    if windhgt != 100:
        windspeed_100m = wind_power_input[windspeed_name]*(100/windhgt)**(1/7)
        print('corrected wind height')
    else:
        windspeed_100m = wind_power_input[windspeed100m]
        
    surface_pressure = wind_power_input[ps]
    surface_spchumidity = wind_power_input[huss]

    if powercurve is None:
        raise ValueError("Please specify power curve")

    GAS_CONSTANT = 287.058  # J/Kg/K

    ##############
    # Calculations
    # Dry air density using ideal gas law
    # rho_d
    dryair_density = surface_pressure / (GAS_CONSTANT * surface_temp)

    # Correct density for humidity
    # rho_m
    corrected_density = dryair_density * (1 + surface_spchumidity) \
                        / (1 + 1.609 * surface_spchumidity)

    # Wind speed scaling for density
    # W_100
    corrected_windspeed_100m = windspeed_100m \
                               * (corrected_density / 1.225) ** (1 / 3)

    return xr.apply_ufunc(powercurve,corrected_windspeed_100m,output_dtypes=[float])
    # return xr.apply_ufunc(powercurve,corrected_windspeed_100m,dask="parallelized",output_dtypes=[float])
