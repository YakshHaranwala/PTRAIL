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

    | Authors: Yaksh J Haranwala
"""
import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures as kin
from ptrail.features.temporal_features import TemporalFeatures as temp

import plotly.express as px


class StatViz:
    @staticmethod
    def trajectory_distance_treemap(dataset: PTRAILDataFrame, path: list):
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
                distance = kin.get_distance_travelled_by_traj_id(dataframe=dataset, traj_id=val)
                duration = temp.get_traj_duration(dataframe=dataset, traj_id=val)
                dist_df.loc[val] = distance / int(duration.dt.days)

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

        palette = ['#B42F32', '#DF6747', '#E3E3CD', '#878D92', '#49494D']
        # Draw the treemap using plotly.
        tree_map = px.treemap(data_frame=dist_df, values='distance', path=path,
                              color_discrete_sequence=palette, title="Average Distance Travelled Per Day")

        # Arrange the margins.
        tree_map.update_layout(margin=dict(t=50, l=25, r=25, b=25))

        # Set the color of the root of the treemap.
        tree_map.update_traces(root_color=palette[3])

        return tree_map
