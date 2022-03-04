# -*- coding: utf-8 -*-

import numpy as np
import math

# Grab ISS position
# Skyfield initialisation
from skyfield.api import load, wgs84

from skyfield.functions import to_spherical
from datetime import timedelta

# Asynchronous and parellel requests
import aiohttp
from aiomultiprocess import Pool
import multiprocessing as mp

# Import helpers
from src.utils.misc import (
    haversine,
    append_zero,
    convert_to_utc_string,
    deriv,
    propagate,
)
from src.utils.satellite import analyse_state_vectors
from src.controllers.leolabs.requests import make_request

async def retrieve_close_approaches(
    state_vectors, location, threshold, target, time=86400
):
    """
    Identifies when satellites enter and exit the observing area, the time and distance of closes approach
    As well as whether the satellite is sunlit within the window
    
    Provide state vectors for satellites from LeoLabs API, an observing location, an observing angular area (threshold)
    And an observing target which is a right ascension and declination
    """

    ts = load.timescale()
    responses = []

    async with Pool(mp.cpu_count()) as pool:
        states = state_vectors["states"]
        
        arguments_map = []
        for state in states:
            arguments = {
                "state_vector": state,
                "location": location,
                "threshold": threshold,
                "target": target,
                "time": time,
                "ts": ts,
            }
            arguments_map.append(arguments)
        
        
        async for response in pool.map(
            analyse_state_vectors,
            arguments_map
        ):
            if response:
                responses.append(response)
            
    return responses

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
    threshold = 5

    # Retrieve state vectors for ISS
    url = "https://api.leolabs.space/v1/catalog/objects/L72,L335,L1159,L2669,L3226,L3969,L3972,L4884,L5011,L5429,L6888/states?latest=true"
    state_vectors = await make_request(url)  # , session)
    import timeit

    start = timeit.default_timer()
    first_responses = await retrieve_close_approaches(
        state_vectors, location, threshold, target
    )
    stop = timeit.default_timer()
    
    print('Time: ', stop - start)  