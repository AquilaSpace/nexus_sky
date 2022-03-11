# -*- coding: utf-8 -*-

import logging
from flask import Blueprint, request, jsonify

import src.controllers.identify_satellites as controllers
from src.app.utils.responses import create_response

satellite = Blueprint('item', __name__)

logger = logging.getLogger(__name__)


@satellite.route('/satellite/retrieve_passes/<key>')
async def retrieve_passes(key):
    try:
        
        location = await controllers.get_location(key)
        data = await controllers.retrieve_close_approaches
    except Exception as e:
        logger.error(e)
        return 