# -*- coding: utf-8 -*-

import numpy as np
import math

# Grab ISS position
# Skyfield initialisation
from skyfield.api import load
from skyfield.functions import to_spherical
from datetime import timedelta

# Import helpers
from src.utils.misc import (
    haversine,
)

# Grab ISS position
# Skyfield initialisation
from dateutil import parser


# Ephemeris for determining if sunlit
# Note, place this in app
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

    return state_id, state_catalog, sat_pos, sat_vel, start_time

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

def identify_close_approaches(
    propagation_list, start_time, target, location, ts, step=600, tolerance=60, stop=False
):
    # Identify the periods where the closest approaches occurred and check whether within tolerance degrees

    last_approach = 360
    last_status = "approaching"
    last_time = start_time

    closest_approaches = []

    # When does the satellite enter and exit the tolerance
    enter_and_exit = []
    approached_within_tolerance = False

    for i in range(len(propagation_list)):

        delta = timedelta(seconds=step * i)
        cur_time = start_time + delta

        # Convert to timescale
        t = ts.from_datetime(cur_time)

        # difference vector
        difference = propagation_list[i][:3] - location.at(t).position.m

        # Convert
        _, dec, ra = to_spherical(difference)
        satellite = [ra * 180 / math.pi, dec * 180 / math.pi]

        distance = haversine(target, satellite)

        # print(distance)

        if (
            distance < tolerance
            and not approached_within_tolerance
            and "approaching"
            and stop
        ):
            approached_within_tolerance = True
            enter_and_exit.append(cur_time)

        elif distance > tolerance and last_status == "receding" and stop:
            enter_and_exit.append(cur_time)

        # Haversine distance
        if distance > last_approach and last_status == "approaching":
            # We've swapped
            last_status = "receding"

            if distance < tolerance:  # Does it approach within tolerance?
                closest_approaches.append(
                    [np.array(propagation_list[i][:3]), last_time, cur_time, distance]
                )

            if stop and distance > tolerance:
                break

        elif distance < last_approach and last_status == "receding":
            last_status = "approaching"

        last_approach = distance
        last_time = cur_time

    return closest_approaches, enter_and_exit