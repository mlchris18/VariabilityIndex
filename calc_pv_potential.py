def calc_pv_potential(solar_potential_input, rsds='rsds', tas='tas', surfWind='surfWind'):
    """ Calculate photovoltaic power potential at a site
    Follows same procedure as https://www.nature.com/articles/ncomms10014#Sec6
    Also: https://www.sciencedirect.com/science/article/pii/S0360544206003501
    Multiply with installed capacity to get power output.

    In:
    rsds
    tas
    wind speed at surface

    Out:

    """
    # Convert inputs to required variable names
    surface_rsds = solar_potential_input[rsds]
    surface_temp = solar_potential_input[tas]
    # The calculations require temperature in Celcius
    if surface_temp.attrs['units'] != 'deg C':
        surface_temp = surface_temp - 273.15

    surface_windspeed = solar_potential_input[surfWind]

    RSDS_STC = 1000.0  # W/m^2
    TEMP_STC = 25.0  # celcius
    GAMMA = -0.005  # /celcius

    ##############
    # Calculations
    # T_cell
    C1 = 4.3
    C2 = 0.943
    C3 = 0.028
    C4 = -1.528
    cell_temp = (C1 + C2 * surface_temp
                 + C3 * surface_rsds + C4 * surface_windspeed)

    # P_R
    performance_ratio = 1.0 + GAMMA * (cell_temp - TEMP_STC)

    # PV_pot
    pv_potential = performance_ratio * surface_rsds / RSDS_STC

    return pv_potential
