"""
    <constants.py>

    Contains all the default constants needed for initialization.
    All the constant are of the type string.
"""

# ------------------------------------- Column-header Constants ---------------------------------#
LAT = 'lat'
LONG = 'lon'
DateTime = 'DateTime'
TRAJECTORY_ID = 'traj_id'
OBJECT_ID = 'object_id'

MANDATORY_COLUMNS = [LAT, LONG, DateTime, TRAJECTORY_ID]

# ----------------------------------- Temporal Constants ---------------------------------------#
WEEKEND = ['Saturday', 'Sunday']

TIME_OF_DAY = {
    0: 'Night',
    1: 'Late Night',
    2: 'Late Night',
    3: 'Late Night',
    4: 'Late Night',
    5: 'Early Morning',
    6: 'Early Morning',
    7: 'Early Morning',
    8: 'Early Morning',
    9: 'Morning',
    10: 'Morning',
    11: 'Morning',
    12: 'Morning',
    13: 'Noon',
    14: 'Noon',
    15: 'Noon',
    16: 'Noon',
    17: 'Evening',
    18: 'Evening',
    19: 'Evening',
    20: 'Evening',
    21: 'Night',
    22: 'Night',
    23: 'Night'
}
# ---------------------------------- Spatial Constants -------------------------------------------#
RADIUS_OF_EARTH = 6371  # KM
