"""
    The semantic features module contains several semantic features like
    intersection of trajectories, stop and stay point detection. Moreover,
    features like distance from Point of Interests, water bodies and other
    demographic features related to the trajectory data are calculated. The
    demographic features are extracted with the help of the python osmnx
    library.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
import itertools
import multiprocessing
import os
from json import JSONDecodeError
from math import ceil
from typing import Union, Text

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.helper_functions import Helpers
from ptrail.utilities.DistanceCalculator import FormulaLog

num = os.cpu_count()
NUM_CPU = ceil((num * 2) / 3)


class SemanticFeatures:
    @staticmethod
    def visited_location(df: PTRAILDataFrame,
                         geo_layers: Union[pd.DataFrame, gpd.GeoDataFrame],
                         visited_location_name: Text,
                         location_column_name: Text):
        """
            Create a column called visited_Location for all the pastures present in the
            dataset.

            Warning
            -------
                While using this method, make sure that the geo_layers parameter dataframe
                that is being passed into the method has Latitude and Longitude columns with
                columns named as 'lat' and 'lon' respectively. If this format is not followed
                then a KeyError will be thrown.

            Note
            ----
                It is to be noted that depending on the size of the dataset and the surrounding
                data passed in, this function will take longer time to execute if either of the
                datasets is very large. It has been parallelized to make it faster, however, it
                can still take a longer time depending on the size of the data being analyzed.

            Parameters
            ----------
                df: PTRAILDataFrame
                    The dataframe containing the dataset.
                geo_layers: Union[pd.DataFrame, gpd.GeoDataFrame]
                    The Dataframe containing the geographical layers near the trajectory data.
                    It is to be noted
                visited_location_name: Text
                    The location for which it is to be checked whether the objected visited it
                    or not.
                location_column_name: Text
                    The name of the column that contains the location to be checked.

            Returns
            -------
                PTRAILDataFrame:
                    The Dataframe containing a new column indicating whether the animal
                    has visited the pasture or not.

        """
        df = df.reset_index()

        # First, filter out the geo_layers dataset to include only the points of the location
        # specified by the user.
        geo_layers = geo_layers.loc[geo_layers[location_column_name] == visited_location_name]

        # Now for the trajectory dataset and the geo layers dataset both, convert them to a
        # GeoDataFrame with the geometry of lat-lon for each point.
        df1 = gpd.GeoDataFrame(df,
                               geometry=gpd.points_from_xy(df["lon"],
                                                           df["lat"]),
                               crs={"init": "epsg:4326"})

        df2 = gpd.GeoDataFrame(geo_layers,
                               geometry=gpd.points_from_xy(geo_layers["lon"],
                                                           geo_layers["lat"]),
                               crs={"init": "epsg:4326"})

        # Now, using GeoPandas, find where the trajectory points and the geo-layers
        # point intersect.
        intersection = gpd.overlay(df1, df2, how='intersection')

        # Now, in the original dataframe, check which points have intersected
        # with the geo-layers dataset and which ones have not.
        merged = pd.merge(df, intersection, how='outer', indicator=True)['_merge']

        # Finally, replace the truth value of the points that have intersected to 1
        # and set it to 0 for the points that have not intersected.
        merged = merged.replace('both', 1)
        merged = merged.replace('left_only', 0)
        merged = merged.replace('right_only', 0)

        # Assign the resultant column to the original df and drop the unnecessary column
        # of geometry.
        df[f'Visited_{visited_location_name}'] = merged
        df = df.drop(columns='geometry')

        # return merged
        return PTRAILDataFrame(df,
                               latitude='lat',
                               longitude='lon',
                               datetime='DateTime',
                               traj_id='traj_id')

    @staticmethod
    def visited_poi(df: PTRAILDataFrame,
                    surrounding_data: Union[gpd.GeoDataFrame, pd.DataFrame, PTRAILDataFrame],
                    dist_column_label: Text,
                    nearby_threshold: int):
        """
            Given a surrounding data with information about the distance to the nearest POI source
            from a given coordinate, check whether the objects in the given trajectory data have
            visited/crossed those POIs or not

            Warning
            -------
                It is to be noted that for this method to work, the surrounding dataset NEEDS to have a
                column containing distance to the nearest POI. For more info, see the Starkey habitat
                dataset which has the columns like 'DistCWat' and 'DistEWat'.


            Parameters
            ----------
                df: PTRAILDataFrame
                    The dataframe containing the trajectory data.
                surrounding_data: Union[gpd.GeoDataFrame, pd.DataFrame]
                    The surrounding data that needs to contain the information of distance
                    to the nearest water body.
                dist_column_label: Text
                    The name of the column containing the distance information.
                nearby_threshold: int
                    The maximum distance between the POI and the current location of the object
                    within which the object is considered to be crossing/visiting the POI.

            Returns
            -------
                PTRAILDataFrame:
                    The dataframe containing the new column indicating whether the object
                    at that point is nearby the POI.

        """
        df_chunks = Helpers._df_split_helper(df)
        print(len(df_chunks))

        # Here, create 2/3rds number of processes as there are CPUs in the system. Some CPUs are
        # kept free at all times in order to not block up the system.
        # (Note: The blocking of system is mostly prevalent in Windows and does not happen very often
        # in Linux. However, out of caution some CPUs are kept free regardless of the system.)
        mp_pool = multiprocessing.Pool(NUM_CPU)
        results = mp_pool.starmap(Helpers.visited_poi_helper,
                                  zip(df_chunks,
                                      itertools.repeat(surrounding_data),
                                      itertools.repeat(dist_column_label),
                                      itertools.repeat(nearby_threshold)
                                      )
                                  )

        mp_pool.close()
        mp_pool.join()

        # Concatenate all the smaller dataframes and return the answer.
        results = pd.concat(results)
        return results

    @staticmethod
    def trajectories_inside_polygon(df: PTRAILDataFrame, polygon: Polygon):
        """
            Given a trajectory dataframe and a Polygon, find out all the trajectories
            that are inside the given polygon.

            Warning
            -------
                While creating a polygon, the format of the coordinates is: (longitude, latitude)
                instead of (latitude, longitude). Beware of that, otherwise the results will be
                incorrect.

            Parameters
            ----------
                df: PTRAILDataFrame
                    The dataframe containing the trajectory data.
                polygon: Polygon
                    The polygon inside which the points are to be found.

            Returns
            -------
                PTRAILDataFrame:
                    A dataframe containing trajectories that are inside the polygon.
        """
        # Convert the original dataframe and the polygon to a GeoPandasDataframe.
        df1 = gpd.GeoDataFrame(df.reset_index(),
                               geometry=gpd.points_from_xy(df["lon"],
                                                           df["lat"]),
                               crs={"init": "epsg:4326"})

        df2 = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygon),
                               crs={"init": "epsg:4326"})

        # Find the points on intersections between the 2 DataFrames.
        intersection = gpd.overlay(df1=df1, df2=df2, how='intersection')
        intersection = intersection.drop(columns=['geometry'])

        # Convert the filtered DF back to PTRAILDataFrame and return it.
        return PTRAILDataFrame(data_set=intersection,
                               datetime='DateTime',
                               traj_id='traj_id',
                               latitude='lat',
                               longitude='lon')

    @staticmethod
    def traj_intersect_inside_polygon(df1: PTRAILDataFrame,
                                      df2: PTRAILDataFrame,
                                      polygon: Polygon):
        """
            Given a df1 and df2 containing trajectory data along with  polygon,
            check whether the trajectory/trajectories are inside the polygon
            and if they are, whether the intersect at any point or not.

            Warning
            -------
                While creating a polygon, the format of the coordinates is: (longitude, latitude)
                instead of (latitude, longitude). Beware of that, otherwise the results will be
                incorrect.

            Note
            ----
                It is to be noted that df1 and df2 should only contain trajectory
                data of only one trajectory each. If they contain more than one
                trajectories, then the results might be unexpected.

            Parameters
            ----------
                df1: PTRAILDataFrame
                    Trajectory Dataframe 1.
                df2: PTRAILDataFrame
                    Trajectory Dataframe 2.
                polygon: Polygon
                    The area inside which it is to be determined if the trajectories
                    intersect or not.

            Returns
            -------
                PTRAILDataFrame:
                    A dataframe containing trajectories that are inside the polygon.
                geopandas.GeoDataFrame:
                    An empty dataframe if both the trajectories do not intersect.
        """
        df1, df2 = df1.reset_index(), df2.reset_index()
        # Convert df1 to GeoDataFrame with correct geometry and the correct CRS.
        gpd_one = gpd.GeoDataFrame(df1,
                                   geometry=gpd.points_from_xy(df1["lon"],
                                                               df1["lat"]),
                                   crs={"init": "epsg:4326"})
        # Convert df2 to GeoDataFrame with correct geometry and the correct CRS.
        gpd_two = gpd.GeoDataFrame(df2,
                                   geometry=gpd.points_from_xy(df2["lon"],
                                                               df2["lat"]),
                                   crs={"init": "epsg:4326"})

        # Convert the polygon to a GeoDataFrame.
        poly_gpd = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygon),
                                    crs={"init": "epsg:4326"})

        # First, checking intersection between t1 and polygon.
        inter_one = gpd.overlay(gpd_one, poly_gpd, 'intersection')

        # Now, checking intersection between t2 and polygon.
        inter_two = gpd.overlay(gpd_two, poly_gpd, 'intersection')

        # Now, checking intersection between inter t1 and t2.
        final = gpd.overlay(inter_one, inter_two, 'intersection')

        if len(final) > 0:
            to_return = final[['traj_id_1', 'traj_id_2', 'lat_1', 'lon_1', 'DateTime_1', 'DateTime_2']]
            to_return = to_return.rename(columns={'lat_1': 'lat',
                                                  'lon_1': 'lon'})
            cols = ['traj_id_1', 'DateTime_1', 'lat', 'lon', 'traj_id_2', 'DateTime_2']
            to_return = to_return[cols]
            return to_return
        else:
            return final

    @staticmethod
    def nearest_poi(coords: tuple, dist_threshold, tags: dict):
        """
            Given a coordinate point and a distance threshold, find the Point of Interest
            which is the nearest to it within the given distance threshold.


            Warning
            -------
                The users are advised the be mindful about the tags being passed in as
                parameter. More the number of tags, longer will the OSMNx library take
                to download the information from the OpenStreetNetwork maps. Moreover,
                an active internet connection is also required to execute this function.

            Note
            ----
                If several tags (POIs) are given in, then the method will find the closest
                one based on the distance and return it and will not given out the others
                that may or may not be present within the threshold of the given point.

            Parameters
            ----------
                coords: tuple
                    The point near which the bank is to be found.
                dist_threshold:
                    The maximum distance from the point within which
                    the distance is to be calculated.
                tags: dict
                    The dictionary containing tags of Points of interest.

            Returns
            -------
                pandas.core.dataframe.DataFrame:
                    A pandas DF containing the info about the nearest bank from
                    the given point.

            Raises
            ------
                JSONDecodeError:
                    One or more given tags are invalid.
        """
        try:
            # Given the tag, the point and the distance threshold, use the osmnx
            # library the fetch the nearby POIs.
            poi = ox.geometries_from_point(center_point=coords,
                                           dist=dist_threshold,
                                           tags=tags)

            # Check whether there are any errors in the tag or if there are actually
            # no point of interests with the given tags nearby the given point.
            if len(poi) > 0:
                # Remove unnecessary column of element_type.
                poi = poi.reset_index().loc[poi.reset_index()['element_type'] == 'node']

                # Convert the Geometry to lat and lon.
                lat = list(poi['geometry'].apply(lambda p: p.y))
                lon = list(poi['geometry'].apply(lambda p: p.x))

                # Check the distances from all nearby resultant POIs and then find out the
                # one which is the nearest to the given point. Mind that the distance
                # calculated is the haversine distance and not walking/driving distance.
                dists = []
                for i in range(len(lat)):
                    dists.append(FormulaLog.haversine_distance(coords[0], coords[1], lat[i], lon[i]))

                poi[f'Distance_from_{coords}'] = dists

                # Return the nearest POI.
                return poi.loc[poi[f'Distance_from_{coords}'] ==
                               poi[f'Distance_from_{coords}'].min()].reset_index().drop(columns=['element_type',
                                                                                                 'index'])
            else:
                # If there are no POIs near the given point, return an empty list.
                return []

        except JSONDecodeError:
            raise ValueError("The tags provided are invalid. Please check your tags and try again.")

    # @staticmethod
    # def detect_herd(df: PTRAILDataFrame,
    #                 surroundings: Union[pd.DataFrame, gpd.GeoDataFrame],
    #                 time_threshold: int):
    #     """
    #         Given a Trajectory Dataframe and another dataframe containing surrounding data,
    #         detect a herd behaviour in a particular region of the surroundings.
    #
    #         Note
    #         ----
    #             The time_threshold is given in seconds.
    #
    #         Parameters
    #         ----------
    #             df: PTRAILDataFrame
    #                 The dataframe containing Trajectory Data.
    #             surroundings: Union[pd.DataFrame, gpd.GeoDataFrame]
    #                 The dataframe containing surrounding data.
    #             time_threshold: int
    #                 The maximum time between 2 points of intersection lesser than which
    #                 the objects are considered to be in herd.
    #     """
    #     # coords = list(zip(surroundings['lon'], (surroundings['lat'])))
    #     # poly = Polygon(coords)
    #     #
    #     # # First, check which of the trajectories are inside the polygon.
    #     # inside_polygon = SemanticFeatures.trajectories_inside_polygon(df, poly)
    #     #
    #     # return inside_polygon
    #
    #     # First split the dataframes based on set of Trajectory ids.
    #     # As of now, each smaller chunk is supposed to have data of 100
    #     # trajectory IDs max
    #     ids_ = list(df.traj_id.value_counts().keys())
    #
    #     df_chunks = [df.loc[df.index.get_level_values(const.TRAJECTORY_ID) == ids_[i]]
    #                  for i in range(len(ids_))]
    #
    #     main = []
    #     for i in range(len(df_chunks) - 1):
    #         df1 = df_chunks[i]
    #         df2 = df_chunks[i + 1]
    #         intersect = SemanticFeatures.traj_intersect_inside_polygon(df1, df2, poly)
    #
    #         time_del = (np.abs((intersect['DateTime_2'] - intersect['DateTime_1'])).dt.total_seconds()).to_numpy()
    #
    #         indices = np.argwhere(time_del <= time_threshold).flatten()
    #         print(indices)
    #
    #         filt_df = intersect.iloc[indices]
    #         main.append(filt_df)
    #
    #     return main
