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
url = "https://api.leolabs.space/v1/catalog/objects/L72,L335,L1159,L2669,L3226,L3969,L3972,L4884,L5011,L5429,L6888"

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

objs = count_objects(resp, 0)
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

# I think we should do a phased approach

# So obtain a state vector and crudely propagate it for the next 24 hours in 1 minute intervals
# Determine if altitude goes above 0 at any point during the observing window
# Filter out satellites that do not

# For the remaining satellites, request one minute propagations over the period they are above the horizon
# Search for the two propagation points that are closest to the target
# Fill in the interval crudely in seconds

# Identify a period in which the satellite centroid is within the tolerance + confidence_interval, otherwise filter
# If occultation is set to False, check if satellite is sunlit within this period and filter out if not. Otherwise note as an occultation
# Request propagations for this period every second and linearly interpolate between points
# Identify and note closest approach (through interpolation). Identify period where the satellite centroid is in tolerance
# Calculate how much of the RMS interval is contained within tolerance, 
# if above threshold note it as interference with associated closest approach and enter/exit times +- given confidence interval

# What does the product look like?
# API with public/secret key

# Request with paramsNow for observational laboratories and something like the Hubble telescope, they use additional factors, including image processing mechanisms that help reduce noise in the image and integration time of the object you are viewing. This means using a sensor that can capture the image at low light levels AND while it moves.

"""
First Route:

find_satellites() -> (or something)

Based on the key-value pair we assign a longitude and latitude

Parameters:
targets: list of pairs (right-ascension, declination) encoding the observing targets
tolerance: (degrees, arcmin, arcsec, milliarcsec) --- how close satellites can come to the target, default 5 degrees
observing_window: (startTime, endTime) default: current sunset+1h to sunrise-1h based on telescope location 
occultation: default False --- include satellites that are occulting but not sunlit
confidence: default 0.95 --- How confident are we the satellite will cross the field of view?

Returns:
List of satellites, their parameters (LeoLabs), period in which they're within the tolerance level, the closest approach to the target + when that occurs

optimise_observing_schedule() -> (or something)
targets: list of pairs (right-ascension, declination) encoding the observing targets
exposure_times: corresponding exposure periods
continuous: default True --- whether the exposures must be continuous
occultation: default False --- prevent occultations from occuring
tolerance: default 10 degrees --- how close can the satellite come to the target before we need to notify you

Returns:
1. Observing schedule that maximises the cumulative distance from the targets to sunlit satellites (and optionally prevents occultations)
2. List of sunlit satellites that come within a set tolerance level to the target, their closest approach and corresponding times

"""


# Product idea. Generate the catalog and attach a peak magnitude to every object therein.
# Astronomers can retrieve the information via API, and send in their data to be cleaned as well

# Provide an API. Generate a key for each observatory which encodes it's location (lat, lon, elevation)
# The key will be regenerated and resent to the observatory operator annually 



