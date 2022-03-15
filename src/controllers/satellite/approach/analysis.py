# -*- coding: utf-8 -*-

import numpy as np
import math

# Grab ISS position
# Skyfield initialisation

from skyfield.functions import to_spherical
from datetime import timedelta

# Import helpers
from src.utils.misc import (
    haversine,
    convert_to_utc_string,
    propagate,
)

# Grab ISS position
# Skyfield initialisation
from dateutil import parser

from src.controllers.leolabs.requests import make_request
from src.utils.satellite import is_sunlit, identify_close_approaches

async def analyse_approach(arguments):
    """
    Analyse a close approach and return its details
    Arguments is a dictionary containing the keys and values
    1. close_approach: A list
    2. location: wgs-84 location
    3. threshold: float
    4. target: tuple
    5. time: timestamp
    6. ts: timescale object
    7. state_id: string
    8. state_catalog: string
    9. sat_pos: array
    10. sat_vel: array
    11. state_time: time
    """
    catalog = {}
    # Check if on night-time side
    _, is_sunside = is_sunlit(
        arguments["close_approach"][0], arguments["ts"].from_datetime(arguments["close_approach"][1])
    )
    if is_sunside:
        return
    
    startTime = convert_to_utc_string(arguments["close_approach"][1] - timedelta(seconds=120))
    endTime = convert_to_utc_string(arguments["close_approach"][2] + timedelta(seconds=120))

    # Make the api call
    url = f"https://api.leolabs.space/v1/catalog/objects/{arguments['state_catalog']}/states/{arguments['state_id']}/propagations?startTime={startTime}&endTime={endTime}&timestep=30"
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
        # Currently we're just using the radar cross section and hoping its informative, will reform over time
        # Estimate magnitude based on distance
        difference = closest_approach[0] - arguments["location"].at(t).position.m
        # This is the formula for estimating magnitude for satellites in Earth orbit
        mag = (
            -26.7
            - 2.5 * math.log10(object_info["rcs"])
            + 5.0 * math.log10(np.linalg.norm(difference).item())
        )

        if len(enter_and_exit) > 1:
            catalog["enters_observing_area"] = str(enter_and_exit[0])
            catalog["exits_observing_area"] = str(enter_and_exit[1])
        else:
            catalog["begins_close_approach"] = str(min_ts[-3])
            catalog["ends_close_approach"] = str(min_ts[-1])

        catalog["closest_approach"] = str(closest_approach[1])
        catalog["closest_separation"] = str(closest_approach[-1])

        catalog["is_sunlit"] = int(is_sunlit(
            closest_approach[0], arguments["ts"].from_datetime(closest_approach[1])
        )[0])
        if is_sunlit:
            catalog["estimated_magnitude"] = str(mag)
            
    return catalog
