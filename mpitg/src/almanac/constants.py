import numpy as np

# Astronomical constants (SI units)
MU_EARTH = 3.986004418e14 # m^3/s^2
MU_SUN = 1.32712440041e20  # m^3/s^2
MU_SUN_KM = 1.3271244004193938e11  # km^3/s^2
AU = 149_597_870_700.0     # meters
G_SI = 6.67430e-11
FRAME = "ECLIPJ2000"
EPOCH_STR = "2020-01-01T00:00:00"
YEARS = 10
SECONDS_PER_YEAR = 365.25 * 24 * 3600
DAY = 86400.0
RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0

# Body SPICE IDs
BODY_IDS = {
    'Mercury': 199,
    'Venus': 299,
    'Earth': 399,
    'Mars-Barycenter': 4,
    'Mars': 499,
    'Jupiter': 599,
    'Saturn': 699,
    'Uranus': 799,
    'Neptune': 899
}


body_names = [
    "SUN", 
    "MERCURY", 
    "VENUS", 
    "EARTH", 
    "MARS",
    "JUPITER BARYCENTER", 
    "SATURN BARYCENTER", 
    "URANUS BARYCENTER", 
    "NEPTUNE BARYCENTER"
]


masses_kg = {
    "SUN": 1.9885e30,
    "MERCURY": 3.3011e23,
    "VENUS": 4.8675e24,
    "EARTH": 5.9722e24,
    "MARS": 6.4171e23,
    "JUPITER": 1.8982e27,
    "SATURN": 5.6834e26,
    "URANUS": 8.6810e25,
    "NEPTUNE": 1.0241e26
}


colors = {
    'Mercury': 'gray',
    'Venus': 'gold',
    'Earth': 'mediumblue',
    'Mars': 'red'
}
