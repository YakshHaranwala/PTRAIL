"""
    | Author: Abhishek Gujjar
"""

import pandas as pd
from typing import Union


def traj_sinuosity_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate TrajSinuosity for a trajectory.

    Args:
        trj: Input trajectory

    Returns:
        TrajSinuosity value or None if an error occurs
    """
    try:
        return TrajSinuosity(trj)
    except:
        return None


# Calculate geometric_sinuosity for each trajectory
geometric_sinuosity = [traj_sinuosity_wrapper(trj) for trj in trajectories]


def traj_sinuosity2_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate TrajSinuosity2 for a trajectory.

    Args:
        trj: Input trajectory

    Returns:
        TrajSinuosity2 value or None if an error occurs
    """
    try:
        return TrajSinuosity2(trj)
    except:
        return None


# Calculate geometric_sinuosity2 for each trajectory
geometric_sinuosity2 = [traj_sinuosity2_wrapper(trj) for trj in trajectories]

# Concatenate traj_features2 DataFrame with geometric_sinuosity and geometric_sinuosity2
traj_features3 = pd.concat([traj_features2, pd.Series(
    geometric_sinuosity), pd.Series(geometric_sinuosity2)], axis=1)


def traj_distance_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate TrajDistance for a trajectory.

    Args:
        trj: Input trajectory

    Returns:
        TrajDistance value or None if an error occurs
    """
    try:
        return TrajDistance(trj)
    except:
        return None


def traj_duration_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate TrajDuration for a trajectory.

    Args:
        trj: Input trajectory

    Returns:
        TrajDuration value or None if an error occurs
    """
    try:
        return TrajDuration(trj)
    except:
        return None


def traj_emax_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate TrajEmax for a trajectory.

    Args:
        trj: Input trajectory

    Returns:
        TrajEmax value or None if an error occurs
    """
    try:
        return TrajEmax(trj)
    except:
        return None


def traj_expected_square_displacement_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate TrajExpectedSquareDisplacement for a trajectory.

    Args:
        trj: Input trajectory

    Returns:
        TrajExpectedSquareDisplacement value or None if an error occurs
    """
    try:
        return TrajExpectedSquareDisplacement(trj)
    except:
        return None


def traj_fractal_dimension_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate TrajFractalDimension for a trajectory.

    Args:
        trj: Input trajectory

    Returns:
        TrajFractalDimension value or None if an error occurs
    """
    try:
        return TrajFractalDimension(trj)
    except:
        return None


def traj_length_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate TrajLength for a trajectory.

    Args:
        trj: Input trajectory

    Returns:
        TrajLength value or None if an error occurs
    """
    try:
        return TrajLength(trj)
    except:
        return None


def traj_straightness_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate TrajStraightness for a trajectory.

    Args:
        trj: Input trajectory

    Returns:
        TrajStraightness value or None if an error occurs
    """
    try:
        return TrajStraightness(trj)
    except:
        return None


# Calculate geometric values for each trajectory
geometric_distance = [traj_distance_wrapper(trj) for trj in trajectories]
geometric_duration = [traj_duration_wrapper(trj) for trj in trajectories]
geometric_emax = [traj_emax_wrapper(trj) for trj in trajectories]
geometric_expected_square_displacement = [
    traj_expected_square_displacement_wrapper(trj) for trj in trajectories]
# geometric_fractal_dimension = [traj_fractal_dimension_wrapper(trj) for trj in trajectories]
geometric_length = [traj_length_wrapper(trj) for trj in trajectories]
geometric_straightness = [
    traj_straightness_wrapper(trj) for trj in trajectories]

# Concatenate geometric values with traj_features3 DataFrame
traj_features3 = pd.concat([
    traj_features3,
    pd.Series(geometric_distance),
    pd.Series(geometric_duration),
    pd.Series(geometric_emax),
    pd.Series(geometric_expected_square_displacement),
    pd.Series(geometric_length),
    pd.Series(geometric_straightness)
], axis=1)

# Save traj_features3 DataFrame as a CSV file
traj_features3.to_csv('traj_features3.csv', index=False)
