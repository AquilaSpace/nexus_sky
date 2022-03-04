# -*- coding: utf-8 -*-

import numpy as np
from dateutil import parser
from skyfield.api import load, wgs84

# Ephemeris for determining if sunlit
eph = load("de421.bsp")


def get_state_info(state_vector):
    """
    Get relevant information from satellite state vector
    """
    state_id = state_vector["id"]
    state_catalog = state_vector["catalogNumber"]
    state_recorded = state_vector["timestamp"]
    start_time = parser.parse(state_recorded)

    sat_pos = np.array(state_vector["frames"]["EME2000"]["position"])
    sat_vel = np.array(state_vector["frames"]["EME2000"]["velocity"])

    uncertainty = (
        state_vector["uncertainties"]["rmsPosition"] * 2
    )  # multiply by 2 to be generous

    return state_id, state_catalog, sat_pos, sat_vel, start_time, uncertainty

def is_sunlit(position, time):
    """

    Parameters
    ----------
    position : np array in meters
    time     : timescale object
    Returns
    -------
    True if sunlit, False if not
    """
    # Need to use a VPS cloud like DigitalOcean (to read the file)
    halfpi, pi, twopi = [f*np.pi for f in (0.5, 1, 2)]
    Re                =   6378000.137 # Radius of the earth
    
    Earth   = eph['earth']
    Sun     = eph['sun']

    sunpos = Sun.at(time).position.m
    earthpos = Earth.at(time).position.m
    satpos = earthpos + position
    
    sunearth, sunsat         = earthpos-sunpos, satpos-sunpos
    sunearthnorm, sunsatnorm = [vec/np.sqrt((vec**2).sum(axis=0)) for vec in (sunearth, sunsat)]
    angle = np.arccos((sunearthnorm * sunsatnorm).sum(axis=0))
    
    sunearthdistance = np.sqrt((sunearth**2).sum(axis=0))
    
    sunsatdistance = np.sqrt((sunsat**2).sum(axis=0))
    limbangle        = np.arctan2(Re, sunearthdistance)
    
    sunlit_bool = (angle > limbangle)
    is_sunside = (sunsatdistance < sunearthdistance)

    return sunlit_bool, is_sunside
