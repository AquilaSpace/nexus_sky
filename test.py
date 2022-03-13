# -*- coding: utf-8 -*-

import requests
from requests.structures import CaseInsensitiveDict


import aiohttp


url = "https://api.leolabs.space/v1/catalog/objects/L72,L335,L1159,L2669,L3226,L3969,L3972,L4884,L5011,L5429,L6888"

def make_request(url):
    # Retrieve calibration objects    
    headers = CaseInsensitiveDict()
    # TODO: Keys file
    headers["Authorization"] = "basic dO4NP7pFiBTnwzS-:K083tL4afQfXBL9Ud97sWsVMFKh3fXN_oQStsVkTgHQ"
    
    resp = requests.get(url, headers=headers)
    return resp

resp = make_request(url)
data = resp.json()

# Binary search using skyfield

"""
Counting objects
"""

# When you get access to the LeoLabs API, do this for every satellite
# for 0 meters
url = "https://api.leolabs.space/v1/catalog/objects"

resp = await make_request(url, session)

def count_objects(data, min_rcs):
    count = 0
    lis = []
    for obj in data['objects']:
        if obj.get('rcs'):
            if obj['rcs'] > min_rcs:
                count += 1
                lis.append(obj)
    print(count)
    return lis


objs = count_objects(resp, 0.0)
# TODO: Figure out minimum cross section
# Ok so it looks like 0.1m^2 rcs is a good benchmark for minimum size

earth_rad = 6378000.137

import math
async def calculate_magnitude(objs):
    ar = []
    async with aiohttp.ClientSession() as session:
        for obj in objs:
            # Calculate the distance above Earth
            item = {}
            url = f"https://api.leolabs.space/v1/catalog/objects/{obj['catalogNumber']}/states?latest=true"
            state_vector = await make_request(url, session)
            state_vector = state_vector["states"][0]["frames"]["EME2000"]["position"]
            
            distance = (state_vector[0]**2 + state_vector[1]**2 + state_vector[2]**2)**0.5 
            
            mag = -26.7 - 2.5 * math.log10(obj['rcs']) + 5.0 * math.log10((distance - earth_rad))
            item["catalog_number"] = obj["catalogNumber"]
            item["estimated_magnitude"] = mag
            ar.append(item)
    return ar

satellites = await calculate_magnitude(objs)


import json
json_string = json.dumps(satellites)

with open('data.json', 'w') as outfile:
    json.dump(json_string, outfile)


# Load file

import json
with open('data.json') as json_file:
    data = json.load(json_file)

data_dict = json.loads(data)

st = set()

for obj in data_dict['objects']:
    st.add(obj['rcs']) # Radar cross section (m^2) is an extremely useful metric for astronomy
# Let's assume an albedo of 1 to be extremely generous


# Retrieve state vectors
url = "https://api.leolabs.space/v1/catalog/objects/L72/states?latest=true"
resp = make_request(url)
data = resp.json()

"""apogee
Propagate the state vectors. 
"""

catalogNumber = data['states'][0]['catalogNumber']
stateId = data['states'][0]['id']

url = f"https://api.leolabs.space/v1/catalog/objects/{catalogNumber}/states/{stateId}/propagations?date=2022-02-19&timestep=3600"
resp = make_request(url)
data = resp.json()


"""
Here are some tools for propagating state vectors
"""

import numpy as np
from scipy.integrate import odeint as ODEint

GMe = 3.986E+14
def deriv(X, t):
    x, v = X.reshape(2, -1)
    a    = -GMe * x * ((x**2).sum())**-1.5
    return np.hstack((v, a))

# Example propagation
pos = data['propagation'][0]['position']
vel = data['propagation'][0]['velocity']
X0 = np.array([pos[0],pos[1],pos[2],vel[0],vel[1],vel[2]])
s = np.arange(60 * 60 * 24) # number of seconds in an hour

answer, info = ODEint(deriv, X0, s, full_output=True)


