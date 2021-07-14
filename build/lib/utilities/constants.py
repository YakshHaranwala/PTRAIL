"""
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

# ----------------------------------- Temporal Constants ----------------------------------------#
WEEKEND = ['Saturday', 'Sunday']

TIME_OF_DAY = [
    'Early Morning',
    'Morning',
    'Noon',
    'Evening',
    'Night',
    'Late Night'
]
# ---------------------------------- Spatial Constants -------------------------------------------#
RADIUS_OF_EARTH = 6371  # KM
PREV_DIST = 'Distance_prev_to_curr'

# ---------------------------------- Splitting Constants -----------------------------------------#
MIN_IDS = 100
