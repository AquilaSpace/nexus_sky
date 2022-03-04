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
    convert_to_utc_string,
    propagate,
)
from src.controllers.leolabs.requests import make_request

# Grab ISS position
# Skyfield initialisation
from dateutil import parser


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

async def analyse_state_vectors(arguments):
    """
    Analyse state vectors and add the satellite to the response if it makes an approach within the threshold
    Arguments is a dictionary containing the keys and values:
        1. state_vector: A dictionary
        2. location: wgs-84 location
        3. threshold: float
        4. target: tuple
        5. time: timestamp
        6. ts: timescale object
    """

    # Get the state vector info and propagate it
    state_id, state_catalog, sat_pos, sat_vel, start_time, uncertainty = get_state_info(
        arguments["state_vector"]
    )
    response = {state_catalog: []}

    
    propagation = propagate(sat_pos, sat_vel, step=60, seconds=arguments["time"])
    close_approaches, _ = identify_close_approaches(
        propagation,
        start_time,
        arguments["target"],
        arguments["location"],
        arguments["ts"],
        step=60,
        tolerance=arguments["threshold"] + 10,
    )

    # Could also break into occultation intervals
    # Get precise information about the approaches from LeoLabs
    for close_approach in close_approaches:
        # Check if on night-time side
        _, is_sunside = is_sunlit(
            close_approach[0], arguments["ts"].from_datetime(close_approach[1])
        )
        if is_sunside:
            continue
        
        startTime = convert_to_utc_string(close_approach[1] - timedelta(seconds=120))
        endTime = convert_to_utc_string(close_approach[2] + timedelta(seconds=120))

        # Make the api call
        url = f"https://api.leolabs.space/v1/catalog/objects/{state_catalog}/states/{state_id}/propagations?startTime={startTime}&endTime={endTime}&timestep=30"
        resp = await make_request(url)  # , session
        data = resp["propagation"]

        min_distance = 360
        min_pos = []
        min_vel = []
        min_ts = []
        distances = []
        
        for point in data:

            timestamp = parser.parse(point["timestamp"])
            t = arguments["ts"].from_datetime(timestamp)

            sat_pos = np.array(point["position"])
            sat_vel = np.array(point["velocity"])
            location_pos = arguments["location"].at(t).position.m

            diff = sat_pos - location_pos
            _, dec, ra = to_spherical(diff)
            satellite = [ra * 180 / math.pi, dec * 180 / math.pi]
            distance = haversine(arguments["target"], satellite)

            distances.append(distance)
            min_ts.append(timestamp)
            if distance < min_distance:
                min_distance = distance
                min_pos.append(sat_pos)
                min_vel.append(sat_vel)
            else:
                break

        # Check when it enters & exits the observing area
        new_prop = []
        if min_distance < arguments["threshold"] + 2:
            # Propagate by hundredths of a second
            new_prop = propagate(min_pos[-2], min_vel[-2], seconds=60, step=0.01)

        closest_approach, enter_and_exit = identify_close_approaches(
            new_prop,
            min_ts[-3],
            arguments["target"],
            arguments["location"],
            arguments["ts"],
            step=0.01,
            tolerance=arguments["threshold"],
            stop=True,
        )

        # Make sure this is functional
        if len(closest_approach) > 0:
            closest_approach = closest_approach[0]
            # Instead of this do estimated magnitude
            state_vector_number = arguments["state_vector"]["catalogNumber"]
            url = f"https://api.leolabs.space/v1/catalog/objects/{state_vector_number}"
            object_info = await make_request(url)  # , session)
            # Currently we're just grabbing the radar cross section and hoping its informative
            # Estimate magnitude based on distance
            # Assume that actual cross section is 10x RCS
            difference = closest_approach[0] - arguments["location"].at(t).position.m
            mag = (
                -26.7
                - 2.5 * math.log10(object_info["rcs"])
                + 5.0 * math.log10(np.linalg.norm(difference).item())
            )

            catalog = {}

            if len(enter_and_exit) > 1:
                catalog["enters_observing_area"] = str(enter_and_exit[0])
                catalog["exits_observing_area"] = str(enter_and_exit[1])
            else:
                catalog["begins_close_approach"] = str(min_ts[-3])
                catalog["ends_close_approach"] = str(min_ts[-1])

            catalog["closest_approach"] = str(closest_approach[1])
            catalog["closest_separation"] = str(closest_approach[-1])

            catalog["is_sunlit"] = is_sunlit(
                closest_approach[0], arguments["ts"].from_datetime(closest_approach[1])
            )[0]
            if is_sunlit:
                catalog["estimated_magnitude"] = mag

            response[state_catalog].append(catalog)
    
    if len(response[state_catalog]) > 0:
        return response

