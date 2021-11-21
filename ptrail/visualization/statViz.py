"""
    This File contains static visualizations i.e the ones that do not require the use of
    ipywidgets.

    Warning
    -------
        The visualizations in this module are currently developed with a focus around the
        starkey.csv data as it has been developed as a side project by the developers. It
        will further be integrated into the library as a general class of visualizers in
        the time to come. Some of the visualization types may or may not work with other
        datasets.

    | Authors: Yaksh J Haranwala, Salman Haidri
"""
import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures as kin
import ptrail.utilities.constants as const

import plotly.express as px


class StatViz:
    @staticmethod
    def trajectory_distance_treemap(dataset: PTRAILDataFrame, map_date: str, path: list):
        """
            Plot a treemap of distance travelled by the moving object on a particular
            date.

            Parameters
            ----------
                dataset: PTRAILDataFrame
                    The dataframe containing all the trajectory data.
                map_date: str
                    The date for which the TreeMap is to be plotted.
                path: list
                    The hierarchy of the treemap. This is passed directly into plotly's
                    Treemap API.

            Returns
            -------
                plotly.graph_objects.Figure:
                    Treemap depicting the distance travelled.
        """
        # First obtain all the unique trajectory IDs of the dataset.
        traj_ids = dataset.reset_index()['traj_id'].unique()

        # Now, for each of the traj_id in the list above, calculate the distance
        # travelled by the moving object on that day and store it in a dictionary.
        dist_df = pd.DataFrame(columns=['traj_id', 'distance'])
        for val in traj_ids:
            try:
                distance = kin.distance_travelled_by_date_and_traj_id(dataframe=dataset,
                                                                      date=map_date, traj_id=val)
                dist_df.loc[val] = distance

            except KeyError:
                # If the animal's trajectory is not recorded on the date given in, just skip it.
                continue

        # Drop the extra column that is acting as the traj ID and reset the index
        # and rename the index column to be traj_id.
        dist_df = dist_df.drop(columns=['traj_id']).reset_index().rename(columns={'index': 'traj_id'})

        species = []
        for i in range(len(dist_df)):
            if 'D' in dist_df.iloc[i]['traj_id']:
                species.append('Deer')
            elif 'E' in dist_df.iloc[i]['traj_id']:
                species.append('Elk')
            else:
                species.append('Cattle')

        dist_df['Species'] = species

        # Draw the treemap using plotly.
        tree_map = px.treemap(data_frame=dist_df, values='distance', path=path)

        # Arrange the margins.
        tree_map.update_layout(margin=dict(t=50, l=25, r=25, b=25))

        # Set the color of the root of the treemap.
        tree_map.update_traces(root_color="cornsilk")

        return tree_map


