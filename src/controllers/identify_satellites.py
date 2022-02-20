# -*- coding: utf-8 -*-

# Grab ISS position
import math
from math import radians, cos, sin, asin, sqrt

from datetime import datetime, timezone, timedelta

import requests
from requests.structures import CaseInsensitiveDict

import numpy as np
# Skyfield initialisation
from skyfield.api import load, wgs84

from skyfield.jpllib import SPICESegment

from scipy.integrate import odeint as ODEint
from skyfield.functions import to_spherical
from skyfield.positionlib import ICRF
from skyfield.constants import ERAD

from dateutil import parser
# Note to self, right-ascension corresponds to latitude and declination corresponds to longitude on celestial sphere

# Ephemeris for determining if sunlit
eph = load('de421.bsp')

# Manually initialise ephemeris for sun and earth

# meters to AU
meters_to_au = 6.6845871226706e-12

# Used for calculating azimuth
tau = 6.283185307179586476925287 
GMe = 3.986E+14

def haversine(target, satellite):
    """
    declination and right ascension coordinates for the target and satellite
    datetime.now(timezone.utc)
    Calculate the angular distance between two targets on the celestial sphere 
    """
    # convert decimal degrees to radians 
    dec1, ra1, dec2, ra2 = map(radians, [target[0], target[1], satellite[0], satellite[1]])

    # haversine formula 
    ddec = dec2 - dec1 
    dra = ra2 - ra1 
    a = sin(dra/2)**2 + cos(ra1) * cos(ra2) * sin(ddec/2)**2
    c = 2 * asin(sqrt(a)) 
    return c * 180 / math.pi

def is_sunlit(position, time):
    """

    Parameters
    ----------
    position : np array in meters
    time     : timescale object
    Returns
    -------
    True if sunlit, False if not
    """
    # Need to test if this works on cloud, because it is reading a local file
    pos_au = position * meters_to_au
    pos_au = pos_au.tolist()
    icrf = ICRF(position_au = position, t = time, center=399) # 399 means ICRF
    return icrf.is_sunlit(eph)

def make_request(url):
    # Retrieve calibration objects    atetime.now(timezone.utc)
    headers = CaseInsensitiveDict()
    # TODO: Keys filedef recurse_into_intsc(state_vector):
    
    pass
    headers["Authorization"] = "basic dO4NP7pFiBTnwzS-:K083tL4afQfXBL9Ud97sWsVMFKh3fXN_oQStsVkTgHQ"
    
    resp = requests.get(url, headers=headers)
    return resp

def deriv(X, t):
    x, v = X.reshape(2, -1)
    a    = -GMe * x * ((x**2).sum())**-1.5
    return np.hstack((v, a))

def propagate(pos, vel, seconds, step=30):
    X0 = np.array([pos[0],pos[1],pos[2],vel[0],vel[1],vel[2]])
    times = np.arange(0, seconds, step)
    answer, _ = ODEint(deriv, X0, times, full_output=True)
    return answer

def identify_close_approaches(propagation_list, target, 
                              start_time, location, step=30, tolerance = 30):
    # Identify the periods where the closest approaches occurred and check whether within tolerance
    closest_approaches = []
        
    ts = load.timescale()
    
    last_approach = 1e10
    last_status = "approaching"
    
    for i in range(len(propagation_list)):
        
        delta = timedelta(seconds = step * i)
        cur_time = start_time + delta
        
        # Convert to timescale
        t = ts.from_datetime(cur_time)
        
        # difference vector
        difference = propagation_list[i] - location.at(t)
        
        # Convert 
        _, dec, ra = to_spherical(difference)
        satellite = dec * 180 / math.pi, ra * 180 / math.pi
        
        # Haversine distance
        distance = haversine(target, satellite)
        if distance > last_approach and last_status == "approaching":
            # We've swapped
            last_status = "receding"
            if distance < tolerance: # Does it approach within tolerance?
                closest_approaches.append(cur_time)
        elif distance < last_approach and last_status == "receding":
            last_status = "approaching"
            
    return closest_approaches


# After you've found the closest approaches just query LeoLabs to get a second-by-second report of the approach
# And yeah... go ahead dude


# Retrieve state vectors for ISS
url = "https://api.leolabs.space/v1/catalog/objects/L72/states?latest=true"
resp = make_request(url)
state_vector = resp.json()

state_id = state_vector['states'][0]['id']
state_recorded = state_vector['states'][0]['timestamp']
start_time = parser.parse(state_recorded)

sat_pos = np.array(state_vector['states'][0]['frames']['EME2000']['position'])
sat_vel = np.array(state_vector['states'][0]['frames']['EME2000']['velocity'])
crude_arr = propagate(sat_pos, sat_vel, 86400)
crude_arr = crude_arr[:, :3]

endTime = state_recorded[:8] + '21' + state_recorded[10:]

# Propagate from recorded state
url = f"https://api.leolabs.space/v1/catalog/objects/L72/states/{state_id}/propagations?startTime={state_recorded}&endTime={endTime}&timestep=3600"
resp = make_request(url)
data = resp.json()

exact_arr = []
for dat in data['propagation']:
    exact_arr.append(dat['position'])

exact_arr  = np.array(exact_arr)


# Start with the Mt Kent Observatory
latitude = -27.7977
longitude = 151.8554
elevation = 682

# We're observing the LMC
# Note, convert to degrees before 
ra_target = 75.39277
dec_target = -69.7561

# You can instead use ts.now() for the current time
ts = load.timescale()
t = ts.now()

mtkent = wgs84.latlon(-27.7977, 151.8554, 682)

# Grab the position in coordinates
coordinates_meters = mtkent.at(t).position.m
difference = sat_pos - coordinates_meters

# Calculate apparent vector
_, dec, ra = to_spherical(difference)

# Convert to degrees
dec, ra = dec * 180 / math.pi, ra * 180 / math.pi


# Calculate difference in propagation
difference = crude_arr[1] - mtkent.at(t).position.m
_, dec, ra = to_spherical(difference)
dec, ra = dec * 180 / math.pi, ra * 180 / math.pi
sat = dec, ra
print(sat)

difference = exact_arr[1] - mtkent.at(t).position.m
_, dec, ra = to_spherical(difference)
dec, ra = dec * 180 / math.pi, ra * 180 / math.pi
target = dec, ra
print(target)

haversine(target, sat)
