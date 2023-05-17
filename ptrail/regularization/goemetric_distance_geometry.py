"""
    | Author: Abhishek Gujjar
"""

import pandas as pd
import numpy as np
from typing import Union, List


def traj_rediscretize_wrapper(trj, step) -> Union[np.ndarray, None]:
    """
    Wrapper function to handle trajectory rediscretization with error handling.

    Args:
        trj: Input trajectory
        step: Rediscretization step

    Returns:
        Rediscretized trajectory or None if an exception occurs.
    """
    try:
        return TrajRediscretize(trj, step)
    except:
        return None


def traj_rediscretize_wrapper_default(trj, step) -> np.ndarray:
    """
    Wrapper function to handle trajectory rediscretization with a default step value if an error occurs.

    Args:
        trj: Input trajectory
        step: Rediscretization step

    Returns:
        Rediscretized trajectory using the provided step or a default step value if an exception occurs.
    """
    try:
        return TrajRediscretize(trj, step)
    except:
        return TrajRediscretize(trj, 0.1)


trajectories_redisc = [traj_rediscretize_wrapper(
    trj, 50) for trj in trajectories]


def extract_subtraj(traj, middle_indices) -> Traj:
    """
    Extracts a subtrajectory from the given trajectory based on the middle indices.

    Args:
        traj: Input trajectory
        middle_indices: List of middle indices for subtrajectory extraction

    Returns:
        Subtrajectory extracted from the original trajectory.
    """
    x_subvector = traj['x'][middle_indices]
    y_subvector = traj['y'][middle_indices]
    coords = pd.DataFrame(
        {'x_subvector': x_subvector, 'y_subvector': y_subvector})
    subtraj = TrajFromCoords(coords)
    return subtraj


def extract_middle_indeces(resampled_traj, no_of_chunks) -> List[int]:
    """
    Extracts middle indices from the resampled trajectory based on the specified number of chunks.

    Args:
        resampled_traj: Resampled trajectory
        no_of_chunks: Number of chunks for extracting middle indices

    Returns:
        List of middle indices extracted from the resampled trajectory.
    """
    indeces = list(range(len(resampled_traj)))
    chunks = chunk(indeces, n_chunks=no_of_chunks)
    mid_indeces = sum(chunks, []) + [len(resampled_traj)]
    return mid_indeces


def cbind_fill(*args) -> pd.DataFrame:
    """
    Combines multiple arrays by column-wise filling and returns a DataFrame.

    Args:
        args: Arrays to be combined

    Returns:
        DataFrame containing the combined arrays with proper column-wise filling.
    """
    nm = [np.asarray(arg) for arg in args]
    n = max(arr.shape[0] for arr in nm)
    return pd.DataFrame(dict(zip(range(len(args)), nm)), index=range(n))


def traj_length_wrapper(trj) -> Union[float, None]:
    """
    Wrapper function to calculate trajectory length with error handling.

    Args:
        trj: Input trajectory

    Returns:
        Trajectory length or None if an exception occurs.
    """
    try:
        return TrajLength(trj)
    except:
        return None


def extract_middle_indeces_wrapper(resampled_traj, no_of_chunks) -> Union[List[int], None]:
    """
    Wrapper function to extract middle indices with error handling.

    Args:
        resampled_traj: Resampled trajectory
        no_of_chunks: Number of chunks for extracting middle indices

    Returns:
        List of middle indices or None if an exception occurs.
    """
    try:
        return extract_middle_indeces(resampled_traj, no_of_chunks)
    except:
        return None


def extract_subtraj_wrapper(traj, middle_indices) -> Union[Traj, None]:
    """
    Wrapper function to extract subtrajectory with error handling.

    Args:
        traj: Input trajectory
        middle_indices: List of middle indices for subtrajectory extraction

    Returns:
        Subtrajectory extracted from the original trajectory or None if an exception occurs.
    """
    try:
        return extract_subtraj(traj, middle_indices)
    except:
        return None


def traj_step_length_wrapper(traj) -> Union[np.ndarray, None]:
    """
    Wrapper function to calculate step lengths of a trajectory with error handling.

    Args:
        traj: Input trajectory

    Returns:
        Array of step lengths or None if an exception occurs.
    """
    try:
        return TrajStepLengths(traj)
    except:
        return None


# Calculate trajectory lengths for each trajectory in trajectories_redisc
traj_lengths = [traj_length_wrapper(trj) for trj in trajectories_redisc]


def adjust_nonzero_values(v: List[float]) -> List[float]:
    """
    Adjust the values greater than 1.0 to 1.0.

    Args:
        v: Input list of values

    Returns:
        List of adjusted values
    """
    adjusted_values = [min(val, 1.0) for val in v]
    return adjusted_values


# Create an empty DataFrame to store distance geometries
distance_geometries = pd.DataFrame()

depth = 5

# Iterate over levels 1 to depth
for level in range(1, depth+1):
    # Get the middle indices for each trajectory at the current level
    mid_indices = [extract_middle_indeces_wrapper(
        trj, level) for trj in trajectories_redisc]

    # Extract sub-trajectories using the middle indices
    sub_trajs = [extract_subtraj_wrapper(trj, indices) for trj, indices in zip(
        trajectories_redisc, mid_indices)]

    # Calculate stepwise distances for each sub-trajectory
    sub_trajs_stepwise_distances = [
        traj_step_length_wrapper(trj) for trj in sub_trajs]

    # Scale trajectory lengths by the current level
    traj_lengths_scaled = [length/level for length in traj_lengths]

    # Normalize stepwise distances by dividing them with the scaled trajectory lengths
    sub_trajs_stepwise_distances_normalized = [
        dist/length if length else None for dist, length in zip(sub_trajs_stepwise_distances, traj_lengths_scaled)]

    # Adjust values greater than 1.0 to 1.0
    sub_trajs_stepwise_distances_normalized = [adjust_nonzero_values(
        dist) if dist else None for dist in sub_trajs_stepwise_distances_normalized]

    # Concatenate the normalized distances horizontally
    new_dg = pd.concat(sub_trajs_stepwise_distances_normalized)

    # Concatenate the new distance geometries to the existing ones vertically
    distance_geometries = pd.concat([distance_geometries, new_dg], axis=1)

    # Clean up variables
    del new_dg, mid_indices, sub_trajs, sub_trajs_stepwise_distances, sub_trajs_stepwise_distances_normalized

# Rename the columns of the distance geometries DataFrame
distance_geometries.columns = ['geometric_distance_1_1', 'geometric_distance_2_1', 'geometric_distance_2_2',
                               'geometric_distance_3_1', 'geometric_distance_3_2', 'geometric_distance_3_3',
                               'geometric_distance_4_1', 'geometric_distance_4_2', 'geometric_distance_4_3',
                               'geometric_distance_4_4', 'geometric_distance_5_1', 'geometric_distance_5_2',
                               'geometric_distance_5_3', 'geometric_distance_5_4', 'geometric_distance_5_5']

# Convert distance_geometries DataFrame to a pickle file
distance_geometries_df = pd.DataFrame(distance_geometries)
distance_geometries_df.to_pickle('distance_geometries_df.pkl')

# Concatenate traj_features DataFrame with distance_geometries_df DataFrame horizontally
traj_features2 = pd.concat([traj_features, distance_geometries_df], axis=1)

# Save traj_features2 DataFrame as a pickle file
traj_features2.to_pickle('traj_features2.pkl')
