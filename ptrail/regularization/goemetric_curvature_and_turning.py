"""
    | Author: Abhishek Gujjar
"""

import numpy as np
from typing import List, Union


def traj_angles_wrapper(trj) -> Union[float, np.nan]:
    """
    Wrapper function to calculate trajectory angles using TrajAngles function.

    Args:
        trj: Input trajectory

    Returns:
        Trajectory angles or NaN if an exception occurs
    """
    try:
        # Calculate trajectory angles using TrajAngles function
        return TrajAngles(trj)
    except Exception as error:
        return np.nan  # If an exception occurs, return NaN


# Calculate angles for each trajectory using traj_angles_wrapper function and store them in a list
angles: List[Union[float, np.nan]] = list(
    map(traj_angles_wrapper, trajectories))

# Calculate the sum of angles for each trajectory, take the absolute value, and store them in a NumPy array
geometric_total_curvature: np.ndarray = np.abs(
    np.array(list(map(sum, angles))))

# Take the absolute value of each angle, calculate the sum for each trajectory, and store them in a NumPy array
geometric_total_turning: np.ndarray = np.array(
    list(map(sum, map(abs, angles))))

# Stack traj_features3, geometric_total_curvature, and geometric_total_turning horizontally and store the result in traj_features4
traj_features4: np.ndarray = np.column_stack(
    (traj_features3, geometric_total_curvature, geometric_total_turning))

# Save traj_features4 as a NumPy binary file named 'traj_features4.npy'
np.save('traj_features4.npy', traj_features4)


# Angle turn counts
# 0 ∼ 45 small angle : 0 - 0.785398
# 45 ∼ 90 medium angle: 0.785398 - 1.5708
# 90 ∼ 135 large angle: 1.5708 - 2.35619
# 135 ∼ 180 reverse angle : 2.35619 - 3.14159

# Calculate angles for each trajectory again and store them in a list
angles: List[Union[float, np.nan]] = list(
    map(traj_angles_wrapper, trajectories))

# Count the number of angles that are within the range (-3.1416, 3.1416) for each trajectory
geometric_angle_turn_count: List[int] = list(
    map(lambda x: np.sum((-3.1416 < x) & (x < 3.1416)), angles))

# Count the number of angles that are less than 0.785398 (small angle) for each trajectory
geometric_small_angle_turn_count: List[int] = list(
    map(lambda x: np.sum(x < 0.785398), angles))

# Count the number of angles that are between 0.785398 (medium angle) and 1.5708 for each trajectory
geometric_medium_angle_turn_count: List[int] = list(
    map(lambda x: np.sum((x > 0.785398) & (x < 1.5708)), angles))

# Count the number of angles that are between 1.5708 (large angle) and 2.35619 for each trajectory
geometric_large_angle_turn_count: List[int] = list(
    map(lambda x: np.sum((x > 1.5708) & (x < 2.35619)), angles))

# Count the number of angles that are greater than 2.35619 (reverse angle) for each trajectory
geometric_reverse_angle_turn_count: List[int] = list(
    map(lambda x: np.sum(x > 2.35619), angles))

# Calculate the proportion of small angles by dividing geometric_small_angle_turn_count by geometric_angle_turn_count
geometric_small_angle_turn_proportion: np.ndarray = np.array
(geometric_small_angle_turn_count) / np.array(geometric_angle_turn_count)

# Calculate the proportion of medium angles by dividing geometric_medium_angle_turn_count
geometric_medium_angle_turn_proportion: np.ndarray = np.array(
    geometric_medium_angle_turn_count) / np.array(geometric_angle_turn_count)
