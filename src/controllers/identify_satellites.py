# -*- coding: utf-8 -*-

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
    Identifies when satellites enter and exit the observing area, the time and distance of closest approach
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
