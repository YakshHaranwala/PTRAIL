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
    'Late Night',
    'Early Morning',
    'Morning',
    'Noon',
    'Evening',
    'Night'
]
# ---------------------------------- Spatial Constants -------------------------------------------#
RADIUS_OF_EARTH = 6371  # KM
PREV_DIST = 'Distance_prev_to_curr'

# ---------------------------------- Splitting Constants -----------------------------------------#
MIN_IDS = 100

# ---------------------------------- Stats Constants --------------------------------------------- #
ORDERED_COLS = [
    '10%_Distance', '25%_Distance', '50%_Distance', '75%_Distance', '90%_Distance', 'min_Distance', 'max_Distance', 'mean_Distance', 'std_Distance',
    '10%_Distance_from_start', '25%_Distance_from_start', '50%_Distance_from_start', '75%_Distance_from_start', '90%_Distance_from_start', 'min_Distance_from_start', 'max_Distance_from_start','mean_Distance_from_start', 'std_Distance_from_start',
    '10%_Speed', '25%_Speed', '50%_Speed', '75%_Speed', '90%_Speed', 'min_Speed', 'max_Speed', 'mean_Speed', 'std_Speed',
    '10%_Acceleration', '25%_Acceleration', '50%_Acceleration', '75%_Acceleration', '90%_Acceleration', 'min_Acceleration', 'max_Acceleration', 'mean_Acceleration', 'std_Acceleration',
    '10%_Jerk', '25%_Jerk', '50%_Jerk', '75%_Jerk', '90%_Jerk', 'min_Jerk', 'max_Jerk', 'mean_Jerk', 'std_Jerk',
    '10%_Bearing', '25%_Bearing', '50%_Bearing', '75%_Bearing', '90%_Bearing', 'min_Bearing', 'max_Bearing', 'mean_Bearing', 'std_Bearing',
    '10%_Bearing_Rate', '25%_Bearing_Rate', '50%_Bearing_Rate', '75%_Bearing_Rate', '90%_Bearing_Rate', 'min_Bearing_Rate','max_Bearing_Rate', 'mean_Bearing_Rate','std_Bearing_Rate',
    '10%_Rate_of_bearing_rate', '25%_Rate_of_bearing_rate', '50%_Rate_of_bearing_rate', '75%_Rate_of_bearing_rate', '90%_Rate_of_bearing_rate', 'min_Rate_of_bearing_rate','max_Rate_of_bearing_rate', 'mean_Rate_of_bearing_rate','std_Rate_of_bearing_rate',
    ]
