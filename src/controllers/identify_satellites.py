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

from astropy.coordinates import SkyCoord

# Note to self, right-ascension corresponds to longitude and declination corresponds to latitude on celestial sphere

# Ephemeris for determining if sunlit
eph = load('de421.bsp')

# Manually initialise ephemeris for sun and earth

# meters to AU
meters_to_au = 6.6845871226706e-12

# Used for calculating azimuth
tau = 6.283185307179586476925287 
GMe = 3.986E+14

def make_request(url):
    # TODO: Keys file
    headers = CaseInsensitiveDict()
    headers["Authorization"] = "basic dO4NP7pFiBTnwzS-:K083tL4afQfXBL9Ud97sWsVMFKh3fXN_oQStsVkTgHQ"
    
    resp = requests.get(url, headers=headers)
    return resp

def haversine(target, satellite):
    """
    declination and right ascension coordinates for the target and satellite
    datetime.now(timezone.utc)
    Calculate the angular distance between two targets on the celestial sphere 
    """
    # convert decimal degrees to radians 
    ra1, dec1, ra2, dec2 = map(radians, [target[0], target[1], satellite[0], satellite[1]])

    # haversine formula 
    dra = ra2 - ra1 
    ddec = dec2 - dec1 
    a = sin(ddec/2)**2 + cos(dec1) * cos(dec2) * sin(dra/2)**2
    c = 2 * asin(sqrt(a)) 
    return c * 180 / math.pi

def append_zero(arr):
    new_arr = []
    for el in arr:
        if len(el) < 2:
            new_arr.append('0' + el)
        else:
            new_arr.append(el)
    return new_arr

def convert_to_utc_string(time):
    month = str(time.month)
    day = str(time.day)   
    hour = str(time.hour)
    minute = str(time.minute)
    second = str(time.second)
    
    arr = append_zero((month, day, hour, minute, second))
            
    return f"{time.year}-{arr[0]}-{arr[1]}T{arr[2]}:{arr[3]}:{arr[4]}Z"

def get_state_info(state_vector):
    state_id  = state_vector['id']
    state_recorded = state_vector['timestamp']
    start_time = parser.parse(state_recorded)

    sat_pos = np.array(state_vector['frames']['EME2000']['position'])
    sat_vel = np.array(state_vector['frames']['EME2000']['velocity'])
    
    uncertainty = state_vector['uncertainties']['rmsPosition'] * 2 # multiply by 2 to be generous
    
    return state_id, sat_pos, sat_vel, start_time, uncertainty

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
    # Need to use a VPS cloud like DigitalOcean (to read the file)
    pos_au = position * meters_to_au
    pos_au = pos_au.tolist()
    icrf = ICRF(position_au = position, t = time, center=399) # 399 means ICRF
    return icrf.is_sunlit(eph)

def deriv(X, t):
    x, v = X.reshape(2, -1)
    a    = -GMe * x * ((x**2).sum())**-1.5
    return np.hstack((v, a))

def propagate(pos, vel, seconds=86400, step=30):
    X0 = np.array([pos[0],pos[1],pos[2],vel[0],vel[1],vel[2]])
    times = np.arange(0, seconds, step)
    answer, _ = ODEint(deriv, X0, times, full_output=True)
    return answer

def identify_close_approaches(propagation_list, start_time, target, location,
                              step = 30, tolerance = 15):
    # Identify the periods where the closest approaches occurred and check whether within tolerance degrees
        
    ts = load.timescale()
    
    last_approach = 360
    last_status = "approaching"
    last_time = start_time
    
    closest_approaches = []
    
    for i in range(len(propagation_list)):
        
        delta = timedelta(seconds = step * i)
        cur_time = start_time + delta
        
        # Convert to timescale
        t = ts.from_datetime(cur_time)
        
        # difference vector
        difference = propagation_list[i][:3] - location.at(t).position.m
        
        # Convert 
        _, dec, ra = to_spherical(difference)
        satellite = [ra * 180 / math.pi, dec * 180 / math.pi]
        
        distance = haversine(target, satellite)
        # Haversine distance
        if distance > last_approach and last_status == "approaching":
            # We've swapped
            last_status = "receding"
            
            if distance < tolerance: # Does it approach within tolerance?
                closest_approaches.append([np.array(propagation_list[i][:3]), cur_time, distance])

        elif distance < last_approach and last_status == "receding":
            last_status = "approaching"
        
        last_approach = distance
            
    return closest_approaches

# After you've found the closest approaches just query LeoLabs to get a second-by-second report of the approach

# Start with the Mt Kent Observatory
latitude = -27.7977
longitude = 151.8554
elevation = 682

location = wgs84.latlon(latitude, longitude, elevation) # Location is mtkent observatory

# We're observing the LMC
# Note, convert to degrees before 
dec_target = -69.7561
ra_target = 75.39277

# And we don't want the satellites to come within 12 degrees of the object

target = [ra_target, dec_target]
threshold = 12

# Retrieve state vectors for ISS
url = "https://api.leolabs.space/v1/catalog/objects/L72,L335,L1159,L2669,L3226,L3969,L3972,L4884,L5011,L5429,L6888/states?latest=true"
resp = make_request(url)
state_vectors = resp.json()


ts = load.timescale()
# Get state vectors
for state_vector in state_vectors['states']:
    # Get the state vector info and propagate it
    state_id, sat_pos, sat_vel, start_time, uncertainty = get_state_info(state_vector)
    propagation = propagate(sat_pos, sat_vel, step=30)
    close_approaches = identify_close_approaches(propagation, start_time, target, 
                                                 location, step=30, tolerance=threshold+10)
    
    # Could also break into occultation intervals
    
    # Get precise information about the approaches from LeoLabs
    for close_approach in close_approaches:
        
        # Add in observing window as well
        if not is_sunlit(close_approach[0], ts.from_datetime(close_approach[1])):
            continue
        
        
        startTime = convert_to_utc_string(close_approach[1] - timedelta(seconds = 120))
        endTime = convert_to_utc_string(close_approach[1] + timedelta(seconds = 120))
        
        # Make the api call
        # Note, need to transition this to asynchronous calls
    
        url = f"https://api.leolabs.space/v1/catalog/objects/L72/states/{state_id}/propagations?startTime={startTime}&endTime={endTime}&timestep=1"
        resp = make_request(url)
        data = resp.json()['propagation']

        for point in data:
            timestamp = parser.parse(point['timestamp'])
            t = ts.from_datetime(timestamp)
            
            satellite_pos  = np.array(point['position'])
            location_pos = location.at(t).position.m
            
            diff = satellite_pos - location_pos
            _, dec, ra = to_spherical(diff)
            satellite = [ra * 180 / math.pi, dec * 180 / math.pi]
            distance = haversine(target, satellite)
            
            if distance < threshold:
                print(timestamp, state_vector['catalogNumber'], distance)
            

        
state_id = state_vectors['states'][0]['id']
state_recorded = state_vectors['states'][0]['timestamp']
start_time = parser.parse(state_recorded)

sat_pos = np.array(state_vectors['states'][0]['frames']['EME2000']['position'])
sat_vel = np.array(state_vectors['states'][0]['frames']['EME2000']['velocity'])


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

# You can instead use ts.now() for the current time
ts = load.timescale()
t = ts.now()

# Grab the position in coordinates
coordinates_meters = location.at(t).position.m
difference = sat_pos - coordinates_meters

# Calculate apparent vector
_, dec, ra = to_spherical(difference)

# Convert to degrees
dec, ra = dec * 180 / math.pi, ra * 180 / math.pi


# Calculate difference in propagation
difference = crude_arr[1] - location.at(t).position.m
_, dec, ra = to_spherical(difference)
dec, ra = dec * 180 / math.pi, ra * 180 / math.pi
sat = dec, ra
print(sat)

# Exact propagation
difference = exact_arr[1] - location.at(t).position.m
_, dec, ra = to_spherical(difference)
dec, ra = dec * 180 / math.pi, ra * 180 / math.pi
target = dec, ra
print(target)

haversine(target, sat)
