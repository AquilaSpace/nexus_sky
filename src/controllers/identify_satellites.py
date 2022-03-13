# -*- coding: utf-8 -*-

# Grab ISS position
# Skyfield initialisation
from skyfield.api import load, wgs84

# Asynchronous and parellel requests
from aiomultiprocess import Pool
import multiprocessing as mp

from src.controllers.satellite.state_vector import analyse_vectors

from src.controllers.leolabs.requests import make_request
from src.controllers.airtable.requests import get_observatories

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
        
        # TODO make multiprocessed 
        for state in states:
            arguments = {
                "state_vector": state,
                "location": location,
                "threshold": threshold,
                "target": target,
                "time": time,
                "ts": ts
            }
            arguments_map.append(arguments)
        
        
        async for response in pool.map(
            analyse_vectors,
            arguments_map
        ):
            if response:
                responses.append(response)
            
    return responses

async def validate_key(key):
    """
    Check a given key is present in the database
    """
    data = await get_observatories()
    records = data['records']
    for observatory in records:
        if observatory['fields']['key'] == key:
            return True
    return False

async def get_location(key):
    """
    Every observatory is assigned a key which is stored in an airtable
    The key encodes its location (latitude, longitude, elevation)
    """
    data = await get_observatories()
    records = data['records']
    for observatory in records:
        if observatory['fields']['key'] == key:
            return wgs84.latlon(observatory['fields']['latitude'], observatory['fields']['longitude'], observatory['fields']['elevation'])
     
    return None

# Test script
if __name__ == "__main__":
    """
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
    threshold = 1

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
    """