"""
    | Author: Abhishek Gujjar
"""
# import rsdepth
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from scipy.spatial import distance
from typing import List, Union


def traj_convexhull_wrapper(pts: np.ndarray) -> Union[np.ndarray, float]:
    """
    Wraps the calculation of the convex hull for a set of points and handles exceptions.

    Args:
        pts (np.ndarray): Array of points representing the trajectory.

    Returns:
        Union[np.ndarray, float]: The convex hull as an array of points if successful,
        otherwise returns NaN if an exception occurs.
    """
    try:
        # return rsdepth.convexhull(pts)
        hull = Polygon(pts).convex_hull
        return np.array(hull.exterior.coords)
    except Exception as error:
        return np.nan


# Calculate convex hulls for each trajectory
chulls = [traj_convexhull_wrapper(np.column_stack(
    (traj['x'], traj['y']))) for traj in trajectories]

# Convert convex hulls to data frames
chull_dfs = [pd.DataFrame(chull, columns=['x', 'y']) for chull in chulls]

# Append the first point of each convex hull to the end to form a closed trajectory
chull_dfs = [pd.concat([df, df.iloc[0]], ignore_index=True)
             for df in chull_dfs]

# Convert convex hull data frames to trajectories
# chull_trajs = [rsdepth.TrajFromCoords(df.values) for df in chull_dfs]

chull_trajs = [Polygon(df.values.tolist()) for df in chull_dfs]

# Calculate geometric_chull_perimeter as the length of each convex hull trajectory
geometric_chull_perimeter = [traj.length() for traj in chull_trajs]

# Combine geometric_chull_perimeter with traj_features5
traj_features6 = pd.concat(
    [traj_features5, pd.Series(geometric_chull_perimeter)], axis=1)

# Calculate centroids of each convex hull
# chulls_centroids = [rsdepth.centroid(chull) for chull in chulls]
chulls_centroids = [Polygon(chull).centroid.coords[0] for chull in chulls]


def major_axis(chull, chull_centroid):
    """
    Calculate the major axes for a convex hull.

    Args:
        chull (np.ndarray): Array of points representing the convex hull.
        chull_centroid (np.ndarray): Array of coordinates representing the centroid of the convex hull.

    Returns:
        np.ndarray: Array of major axes distances for each point in the convex hull.
    """
    try:
        return np.apply_along_axis(lambda vertex: distance.euclidean(vertex, chull_centroid), 1, chull)
    except Exception as error:
        # Handle the exception (e.g., print an error message, return a default value, etc.)
        print("An error occurred:", str(error))
        return np.array([])  # Return an empty array or any other default value


def major_axis_wrapper(chull: np.ndarray, chull_centroid: np.ndarray) -> Union[np.ndarray, float]:
    """
    Wraps the calculation of major axis distances for a convex hull and handles exceptions.

    Args:
        chull (np.ndarray): Array of points representing the convex hull.
        chull_centroid (np.ndarray): Array of coordinates representing the centroid of the convex hull.

    Returns:
        Union[np.ndarray, float]: Array of major axis distances for each point in the convex hull if successful,
        otherwise returns NaN if an exception occurs.
    """
    try:
        return np.apply_along_axis(lambda vertex: np.linalg.norm(vertex - chull_centroid), 1, chull)
    except Exception as error:
        return np.nan


# Calculate major axis distances for each convex hull
chulls_major_axes = [major_axis_wrapper(
    chull, centroid) for chull, centroid in zip(chulls, chulls_centroids)]

# Calculate the maximum major axis distance for each convex hull
chulls_major_axis = [np.max(distances) for distances in chulls_major_axes]

# Shift convex hulls by removing the first point and appending the first point at the end
chulls_shifted = [chull[1:] for chull in chulls]
chulls_shifted = [np.vstack((shifted, chull[0]))
                  for shifted, chull in zip(chulls_shifted, chulls)]

# Convert convex hulls and shifted hulls to data frames
chulls_df = [pd.DataFrame(chull, columns=['x', 'y']) for chull in chulls]
chulls_shifted_df = [pd.DataFrame(shifted, columns=['x', 'y'])
                     for shifted in chulls_shifted]

# Rename columns in chulls_shifted_df
chulls_shifted_df = [
    df.rename(columns={'x': 'V3', 'y': 'V4'}) for df in chulls_shifted_df]

# Combine convex hull data frames and shifted data frames
chulls_traj_bounding_points = [pd.concat(
    [df1, df2], axis=1) for df1, df2 in zip(chulls_df, chulls_shifted_df)]

# Convert centroids to data frames
chulls_centroids_df = [pd.DataFrame(np.transpose(centroid), columns=[
                                    'V5', 'V6']) for centroid in chulls_centroids]

# Rename columns in chulls_centroids_df
chulls_centroids_df = [
    df.rename(columns={'V1': 'V5', 'V2': 'V6'}) for df in chulls_centroids_df]

# Combine chulls_traj_bounding_points and chulls_centroids_df
chulls_traj_bounding_points_and_centroid = [pd.concat(
    [df1, df2], axis=1) for df1, df2 in zip(chulls_traj_bounding_points, chulls_centroids_df)]


def distance_point_to_line_segment(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    """
    Calculates the distance from a point (x3, y3) to a line segment defined by two points (x1, y1) and (x2, y2).

    Args:
        x1 (float): x-coordinate of the first point of the line segment.
        y1 (float): y-coordinate of the first point of the line segment.
        x2 (float): x-coordinate of the second point of the line segment.
        y2 (float): y-coordinate of the second point of the line segment.
        x3 (float): x-coordinate of the point.
        y3 (float): y-coordinate of the point.

    Returns:
        float: The distance from the point to the line segment.
    """
    px = x2 - x1
    py = y2 - y1

    norm = px * px + py * py

    if norm < 0.0000000001:
        norm = 0.0000000001

    u = ((x3 - x1) * px + (y3 - y1) * py) / norm

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = np.sqrt(dx * dx + dy * dy)

    return dist


def distance_point_to_line_segment_wrapper(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> Union[float, None]:
    """
    Wraps the calculation of distance from a point to a line segment and handles exceptions.

    Args:
        x1 (float): x-coordinate of the first point of the line segment.
        y1 (float): y-coordinate of the first point of the line segment.
        x2 (float): x-coordinate of the second point of the line segment.
        y2 (float): y-coordinate of the second point of the line segment.
        x3 (float): x-coordinate of the point.
        y3 (float): y-coordinate of the point.

    Returns:
        Union[float, None]: The distance from the point to the line segment if successful,
        otherwise returns None if an exception occurs.
    """
    try:
        return distance_point_to_line_segment(x1, y1, x2, y2, x3, y3)
    except Exception as error:
        return None


# Calculate chulls_minor_axes using the distance_point_to_line_segment_wrapper function
# Iterate over chulls_traj_bounding_points_and_centroid and apply distance_point_to_line_segment_wrapper on each row
# Store the results as a list of lists
chulls_minor_axes = [list(map(lambda x: distance_point_to_line_segment_wrapper(
    x[0], x[1], x[2], x[3], x[4], x[5]), df.values)) for df in chulls_traj_bounding_points_and_centroid]

# Calculate chulls_minor_axis as the minimum value in each list of distances
chulls_minor_axis = [min(np.concatenate(x)) for x in chulls_minor_axes]


# Calculate the geometric chull aspect ratio
geometric_chull_aspect_ratio = [x / y for x,
                                y in zip(chulls_minor_axis, chulls_major_axis)]

# Combine traj_features6 and geometric_chull_aspect_ratio
traj_features7 = np.concatenate((traj_features6, np.array(
    geometric_chull_aspect_ratio)[:, np.newaxis]), axis=1)

# Calculate chull_areas_list using the Polygon area function from shapely library
chull_areas_list = [Polygon(chull).area for chull in chulls]

# Convert chull_areas_list to a dataframe
chull_areas_df = pd.DataFrame(chull_areas_list)

# Extract the second column of chull_areas_df
chull_areas = chull_areas_df.iloc[:, 1]

# Convert chull_areas to a list
geometric_chull_area = chull_areas.tolist()

# Combine traj_features7 and geometric_chull_area
traj_features8 = np.concatenate(
    (traj_features7, np.array(geometric_chull_area)[:, np.newaxis]), axis=1)

# Calculate chulls using traj_convexhull_wrapper
chulls = [traj_convexhull_wrapper(np.hstack(
    (l['x'][:, np.newaxis], l['y'][:, np.newaxis]))) for l in trajectories]

# Calculate chulls_centroids
# chulls_centroids = [rsdepth.centroid(chull) for chull in chulls]
chulls_centroids = [Polygon(chull).centroid for chull in chulls]

# Calculate chulls_major_axes
chulls_major_axes = [list(map(lambda x: np.argmax(x), major_axis(x, y)))
                     for x, y in zip(chulls, chulls_centroids)]


# Calculate chulls_major_axis
chulls_major_axis = [max(x) for x in chulls_major_axes]

# Convert chulls to dataframes
chulls_df = [pd.DataFrame(chull) for chull in chulls]

# Convert chulls_centroids to a dataframe
chulls_centroids_df = pd.DataFrame(chulls_centroids)

# Extract the corresponding rows from chulls_df based on chulls_major_axis
chulls_major_axis_coords_partial = pd.concat(
    [chull.iloc[major_axis, :] for chull, major_axis in zip(chulls_df, chulls_major_axis)], ignore_index=True)

# Rename columns of chulls_major_axis_coords_partial
chulls_major_axis_coords_partial.rename(
    columns={0: 'V3', 1: 'V4'}, inplace=True)

# Combine chulls_centroids_df and chulls_major_axis_coords_partial
chulls_major_axis_coords = pd.concat(
    [chulls_centroids_df, chulls_major_axis_coords_partial], axis=1)

# Calculate x and y coordinates for geometric chull orientation
x = chulls_major_axis_coords['V3'].astype(
    float) - chulls_major_axis_coords['V1'].astype(float)
y = chulls_major_axis_coords['V4'].astype(
    float) - chulls_major_axis_coords['V2'].astype(float)

# Calculate geometric chull orientation using atan
geometric_chull_orientation = np.arctan2(y, x)

# Combine traj_features8 and geometric_chull_orientation
traj_features6 = np.concatenate((traj_features8, np.array(
    geometric_chull_orientation)[:, np.newaxis]), axis=1)

# Save traj_features6 to file
np.save('traj_features6.npy', traj_features6)
