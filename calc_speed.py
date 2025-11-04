import numpy as np
import xarray as xr


def calc_speed(u_v_dset, eastward_velocity, northward_velocity, attributes=None):
    """ Calculate the wind speed from individual components. Inputs
    can be as numpy array or xarray DataArray of same shape. Output
    will also be of same shape. If the inputs are xarrays,

    In:
    eastward_velocity: Eastward wind component, mostly denoted by
    'uas/ua100m' etc in climate datasets
    northward_velocity: Northward wind component, mostly denoted by
    'vas/va100m' etc in climate datasets

    Returns:
    wind speed

    """
    wind_speed = np.sqrt(u_v_dset[eastward_velocity] ** 2 
                         + u_v_dset[northward_velocity] ** 2)

    if isinstance(u_v_dset[eastward_velocity], xr.DataArray):
        if attributes is None:
            wind_speed.attrs['units'] = u_v_dset[eastward_velocity].attrs['units']
            wind_speed.attrs['comment'] = 'Wind speed'
        else:
            for key, value in attributes.items():
                wind_speed.attrs[key] = value
    return wind_speed
