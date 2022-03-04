# -*- coding: utf-8 -*-

import numpy as np
import math

# Grab ISS position
# Skyfield initialisation
from skyfield.api import load, wgs84

from skyfield.functions import to_spherical
from dateutil import parser

from datetime import timedelta

# Multiprocessing
import multiprocessing as mp

# Asynchronous requests
import aiohttp

# Import helpers
from src.utils.misc import haversine, append_zero, convert_to_utc_string, deriv, propagate
from src.utils.satellite import get_state_info, is_sunlit
from src.controllers.leolabs.requests import make_request

def identify_close_approaches(
    propagation_list, start_time, target, location, step=600, tolerance=60, stop=False
):
    # Identify the periods where the closest approaches occurred and check whether within tolerance degrees

    ts = load.timescale()

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
        
        #print(distance)
        
        if distance < tolerance and not approached_within_tolerance and "approaching" and stop:
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

async def retrieve_close_approaches(state_vectors, location, threshold, target, time=86400):
    """
    Identifies when satellites enter and exit the observing area, the time and distance of closes approach
    As well as whether the satellite is sunlit within the window
    
    Provide state vectors for satellites from LeoLabs API, an observing location, an observing angular area (threshold)
    And an observing target which is a right ascension and declination
    
    Note: need to parallelize
    """
    responses = []
    
    pool = mp.Pool(mp.cpu_count())
    
    async with aiohttp.ClientSession() as session:
        ts = load.timescale()
        # Get state vectors
        for state_vector in state_vectors["states"]:
            # Get the state vector info and propagate it
            state_id, state_catalog, sat_pos, sat_vel, start_time, uncertainty = get_state_info(state_vector)
            propagation = propagate(sat_pos, sat_vel, step=60, seconds=time)
            
            close_approaches, _ = identify_close_approaches(
                propagation, start_time, target, location, step=60, tolerance = threshold + 10
            )
        
            # Could also break into occultation intervals
        
            # Get precise information about the approaches from LeoLabs
            for close_approach in close_approaches:
                # Check if on night-time side
                _, is_sunside = is_sunlit(close_approach[0], ts.from_datetime(close_approach[1]))
                if is_sunside:
                    continue
        
                startTime = convert_to_utc_string(close_approach[1] - timedelta(seconds = 120))
                endTime = convert_to_utc_string(close_approach[2] + timedelta(seconds = 120))
                
                # Logic, check if startTime and endTime are in observing window also check if they are sunlit
        
                # Make the api call
                # Note, need to transition this to asynchronous calls
        
                url = f"https://api.leolabs.space/v1/catalog/objects/{state_catalog}/states/{state_id}/propagations?startTime={startTime}&endTime={endTime}&timestep=30"
                resp = await make_request(url, session)
                data = resp["propagation"]
                
                min_distance = 360
                min_pos = []
                min_vel = []
                min_ts  = []
                distances = []
                
                for point in data:
                    
                    timestamp = parser.parse(point["timestamp"])
                    t = ts.from_datetime(timestamp)
        
                    sat_pos = np.array(point["position"])
                    sat_vel = np.array(point["velocity"])
                    location_pos = location.at(t).position.m
        
                    diff = sat_pos - location_pos
                    _, dec, ra = to_spherical(diff)
                    satellite = [ra * 180 / math.pi, dec * 180 / math.pi]
                    distance = haversine(target, satellite)
                    
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
                if min_distance < threshold + 2:
                    # Propagate by hundredths of a second
                    new_prop = propagate(min_pos[-2], min_vel[-2], seconds=60, step=0.01)
                
                closest_approach, enter_and_exit = identify_close_approaches(
                    new_prop, min_ts[-3], target, location, step=0.01, tolerance = threshold, stop=True
                )
                
                # Make sure this is functional
                if len(closest_approach) > 0:
                    closest_approach = closest_approach[0]
                    # Instead of this do estimated magnitude
                    state_vector_number = state_vector["catalogNumber"]
                    url = f"https://api.leolabs.space/v1/catalog/objects/{state_vector_number}"
                    object_info = await make_request(url, session)
                    # Currently we're just grabbing the radar cross section and hoping its informative
                    # Estimate magnitude based on distance
                    # Assume that actual cross section is 10x RCS
                    difference = closest_approach[0] - location.at(t).position.m
                    mag = -26.7 - 2.5 * math.log10(object_info['rcs']) + 5.0 * math.log10(
                                                                                np.linalg.norm(difference).item())
                    response = {}
                    
                    if len(enter_and_exit) > 1:
                        response["enters_observing_area"] = str(enter_and_exit[0])
                        response["exits_observing_area"] = str(enter_and_exit[1])
                    else:
                        response["begins_close_approach"] = str(min_ts[-3])
                        response["ends_close_approach"] = str(min_ts[-1])
                            

                    response["closest_approach"] = str(closest_approach[1])
                    response["closest_separation"] = str(closest_approach[-1])
                    session
                    response["is_sunlit"] = is_sunlit(closest_approach[0], ts.from_datetime(closest_approach[1]))[0]
                    if is_sunlit:
                        response["estimated_magnitude"] = mag
                    
                    responses.append(response)
    
    return responses
                
# After you've found the closest approaches just query LeoLabs to get a second-by-second report of the approach

# Test script
if __name__ == "__main__":
    # Start with the Mt Kent Observatory
    latitude = -27.7977
    longitude = 151.8554
    elevation = 682
    
    location = wgs84.latlon(
        latitude, longitude, elevation
    )  # Location is mtkent observatory
    
    # We're observing the LMC
    # Note, convert to degrees before
    dec_target = -69.7561
    ra_target = 75.39277
    
    # And we don't want the satellites to come within 2 degrees of the object
    target = [ra_target, dec_target]
    threshold = 20
    
    # Retrieve state vectors for ISS
    url = "https://api.leolabs.space/v1/catalog/objects/L72,L335,L1159,L2669,L3226,L3969,L3972,L4884,L5011,L5429,L6888/states?latest=true"
    state_vectors = await make_request(url, session)
    
    await main(state_vectors, location, threshold, target)
