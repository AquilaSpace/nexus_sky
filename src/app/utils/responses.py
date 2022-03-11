# -*- coding: utf-8 -*-

import json
from flask import jsonify

def create_response(res):
    """
    General abstraction to create desired response
    """
    if isinstance(res, dict):
        if res['status'] == 'failure':
            # This means an error code has been returned 
            code = res['code']
            return jsonify(res), code 
        else:
            return jsonify(res), 200
    # TODO: Change in future
    return jsonify({'status': 'success', 'data': str(res) }), 200

# TODO: Response factory for every method specific verb.
# Flesh out responses as well