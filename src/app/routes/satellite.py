# -*- coding: utf-8 -*-

import logging
from flask import Blueprint, request, jsonify

import src.controllers.identify_satellites as controllers
from src.app.utils.responses import create_response
from src.controllers.leolabs.requests import make_request

from skyfield.api import wgs84

satellite = Blueprint('item', __name__)

logger = logging.getLogger(__name__)

@satellite.route('/satellite/retrieve_passes/<key>', methods=['POST'])
async def retrieve_passes(key):
    data = request.json
    if not data:
        return jsonify({'status': 'failure', 'error': 'please provide a list of \
                        catalog objects (e.g. catalog_objects: "all" or \
                                         catalog_objects: "L72,L335,L1159,L2669,L3226,\
                                             L3969,L3972,L4884,L5011,L5429,L6888)" \
                as well as the celestial position, observing field and optionally a time horizon in your request. \
                    See https://www.aquila.earth/whitepaper for details'})
    try:
        # TODO: migrate over to a cache
        location = await controllers.get_location(key)
        if not location:
            # Return location could not be found
            return jsonify({'status': 'failure', 'error': 'invalid key'})
        
        catalog_objects = data['catalog_objects']
        threshold = data['threshold']
        target = data['target']
        
        time = data.get('time')
        
        
        url = f"https://api.leolabs.space/v1/catalog/objects/{catalog_objects}/states?latest=true"
        state_vectors = await make_request(url)
        
        if time:
            response = await controllers.retrieve_close_approaches(state_vectors, location, 
                                                               threshold, target, time)
        else:
            response = await controllers.retrieve_close_approaches(state_vectors, location, threshold, target, time)
            
    except Exception as e:
        logger.error(e)
        return jsonify({'status': 'failure', 'error': e})
    
    return jsonify({'status': 'success', 'data': response}), 200