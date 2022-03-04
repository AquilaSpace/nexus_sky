# -*- coding: utf-8 -*-


from src.utils.satellite import get_state_info, identify_close_approaches
from src.utils.misc import propagate

from aiomultiprocess import Pool
import multiprocessing as mp

from src.controllers.satellite.approach.analysis import analyse_approach

async def analyse_vectors(arguments):
    """
    Analyse state vectors and add the satellite to the response if it makes an approach within the threshold.
    Arguments is a dictionary containing the keys and values:
        1. state_vector: A dictionary
        2. location: wgs-84 location
        3. threshold: float
        4. target: tuple
        5. time: timestamp
        6. ts: timescale object
    """
    # Get the state vector info and propagate it
    state_id, state_catalog, sat_pos, sat_vel, start_time = get_state_info(
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
        # The new dictionary
        new_arguments = arguments.copy()
        new_arguments["close_approach"] = close_approach
        new_arguments["state_id"] = state_id
        new_arguments["state_catalog"] = state_catalog
        new_arguments["sat_pos"] = sat_pos
        new_arguments["sat_vel"] = sat_vel
        new_arguments["start_time"] = start_time
        
        catalog = await analyse_approach(new_arguments)
        if catalog:
            response[state_catalog].append(catalog)    
        
    if len(response[state_catalog]) > 0:
        return response

