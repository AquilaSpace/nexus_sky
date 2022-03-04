# -*- coding: utf-8 -*-

import math
import numpy as np
from math import radians, cos, sin, asin, sqrt

from scipy.integrate import odeint as ODEint
from skyfield.functions import to_spherical
from skyfield.positionlib import ICRF

import json

# Note to self, right-ascension corresponds to longitude and declination corresponds to latitude on celestial sphere

# Manually initialise ephemeris for sun and earth

# meters to AU
meters_to_au = 6.6845871226706e-12

# Used for calculating azimuth
tau = 6.283185307179586476925287
GMe = 3.986e14

def haversine(target, satellite):
    """
    declination and right ascension coordinates for the target and satellite
    datetime.now(timezone.utc)
    Calculate the angular distance between two targets on the celestial sphere 
    """
    # convert decimal degrees to radians
    ra1, dec1, ra2, dec2 = map(
        radians, [target[0], target[1], satellite[0], satellite[1]]
    )

    # haversine formula
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = sin(ddec / 2) ** 2 + cos(dec1) * cos(dec2) * sin(dra / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 180 / math.pi


def append_zero(arr):
    new_arr = []
    for el in arr:
        if len(el) < 2:
            new_arr.append("0" + el)
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


def deriv(X, t):
    x, v = X.reshape(2, -1)
    a = -GMe * x * ((x ** 2).sum()) ** -1.5
    return np.hstack((v, a))


def propagate(pos, vel, seconds=86400, step=600):
    X0 = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
    times = np.arange(0, seconds, step)
    answer, _ = ODEint(deriv, X0, times, full_output=True)
    return answer



