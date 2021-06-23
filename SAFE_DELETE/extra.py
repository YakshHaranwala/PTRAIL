# @staticmethod
# def _linear_ip(dataframe, time_jump):
#     """
#         Interpolate the position of points using the Linear Interpolation method. It makes
#         the use of numpy's interpolation technique for the interpolation of the points.
#
#         WARNING: Do not use this method directly as it will run slower. Instead,
#                  use the method interpolate_position() and specify the ip_type as
#                  linear to perform linear interpolation much faster.
#
#         Parameters
#         ----------
#             dataframe: NumPandasTraj
#                 The dataframe containing the original data.
#             time_jump: float
#                 The maximum time difference between 2 points. If the time difference between
#                 2 consecutive points is greater than the time jump, then another point will
#                 be inserted between the given 2 points.
#
#         Returns
#         -------
#             pandas.core.dataframe.DataFrame:
#                 The dataframe enhanced with interpolated points.
#     """
#     # First, reset the index, extract the Latitude, Longitude, DateTime and Trajectory ID columns
#     # and set the DateTime column only as the index. Then, store all the unique Trajectory IDs in
#     # a list.
#     dataframe = dataframe.reset_index(drop=True)[
#         [const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)
#     ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())
#
#     # Now, for each unique ID in the dataframe, interpolate the points.
#     for i in range(len(ids_)):
#         df = dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]]   # Extract points of only 1 traj ID.
#         # Create a Series containing new times which are calculated as follows:
#         #    new_time[i] = original_time[i] + time_jump.
#         new_times = df.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')
#
#         # Now, interpolate the latitudes using numpy based on the new times calculated above.
#         ip_lat = np.interp(new_times,
#                            df.reset_index()[const.DateTime],
#                            df.reset_index()[const.LAT])
#
#         # Now, interpolate the longitudes using numpy based on the new times calculated above.
#         ip_long = np.interp(new_times,
#                             df.reset_index()[const.DateTime],
#                             df.reset_index()[const.LONG])
#
#         # Here, store the time difference between all the consecutive points in an array.
#         time_deltas = df.reset_index()[const.DateTime].diff().dt.total_seconds()
#         id_ = df.reset_index()[const.TRAJECTORY_ID].iloc[0]
#
#         # Now, for each point in the trajectory, check whether the time difference between
#         # 2 consecutive points is greater than the user-specified time_jump, and if so then
#         # insert a new point that is linearly interpolated between the 2 original points.
#         for j in range(len(time_deltas)):
#             if time_deltas[j] > time_jump:
#                 dataframe.loc[new_times[j-1]] = [id_, ip_lat[j-1], ip_long[j-1]]
#
#     return dataframe

# @staticmethod
# def _cubic_ip():
# # First, reset the index, extract the Latitude, Longitude, DateTime and Trajectory ID columns
# # and set the DateTime column only as the index. Then, store all the unique Trajectory IDs in
# # a list
# dataframe = dataframe.reset_index(drop=True)[
#     [const.DateTime, const.TRAJECTORY_ID, const.LAT, const.LONG]].set_index(const.DateTime)
# ids_ = list(dataframe[const.TRAJECTORY_ID].value_counts().keys())
#
# # Now, for each unique ID in the dataframe, interpolate the points.
# for i in range(len(ids_)):
#     df = dataframe.loc[dataframe[const.TRAJECTORY_ID] == ids_[i]]
#
#     # If the trajectory has less than 3 points, then skip the trajectory
#     # from the interpolation.
#     if len(df) < 3:
#         continue
#
#     # Create a Series containing new times which are calculated as follows:
#     #    new_time[i] = original_time[i] + time_jump.
#     new_times = df.reset_index()[const.DateTime] + pd.to_timedelta(time_jump, unit='seconds')
#
#     # Extract the Latitude, Longitude pairs for each point and store it in a
#     # numpy array.
#     coords = df.reset_index()[[const.LAT, const.LONG]].to_numpy()
#
#     # Now, using Scipy's Cubic spline, create a spline object for interpolation of
#     # points.
#     cubic_spline = CubicSpline(x=df.reset_index()[const.DateTime],
#                                y=coords,
#                                extrapolate=True, bc_type='not-a-knot')
#
#     # Now, calculate the interpolated position of the points at all the new_times
#     # calculated above.
#     ip_coords = cubic_spline(new_times)
#
#     # Here, store the time difference between all the consecutive points in an array.
#     time_deltas = df.reset_index()[const.DateTime].diff().dt.total_seconds()
#     id_ = df.reset_index()[const.TRAJECTORY_ID].iloc[0]
#
#     # Now, for each point in the trajectory, check whether the time difference between
#     # 2 consecutive points is greater than the user-specified time_jump, and if so then
#     # insert a new point that is cubic-spline interpolated between the 2 original points.
#     for j in range(len(time_deltas)):
#         if time_deltas[j] > time_jump:
#             dataframe.loc[new_times[j - 1]] = [id_, ip_coords[j - 1][0], ip_coords[j - 1][1]]
#
# return dataframe