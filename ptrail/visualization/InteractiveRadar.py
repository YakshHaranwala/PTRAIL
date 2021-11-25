"""
    This File contains the visualization that is a Radar Scatter plot showing how many
    number of days around each Running Water Body has an individual element spent. It
    is an interactive visualization as such the user can change the water body and check
    the distribution of animals around it.

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
from IPython.core.display import display
from ipywidgets import widgets
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.temporal_features import TemporalFeatures as temp
from ptrail.preprocessing.filters import Filters as filt

import plotly.express as px

pd.options.mode.chained_assignment = None


class InteractiveRadar:
    # Class variables to store datasets and the widget.
    __habitat_data = None
    __traj_data = None
    __point_dict = None
    __list = None

    @staticmethod
    def show_distribution_radar(trajectories: PTRAILDataFrame, habitat: pd.DataFrame,
                                dist_from_water: int):
        """
            Plot the interactive plotly Radar chart that shows the number of days spent
            by animals around a specific water body.

            Note
            ----
                The water bodies in the original dataset do not have any specific names.
                Hence, they are just given names such as Water-body #1, Water-body #2
                and so on.

            Parameters
            ----------
                trajectories: PTRAILDataFrame
                    The dataframe containing the trajectory data.
                habitat: pd.DataFrame
                    The dataframe containing the habitat data.
                dist_from_water: int
                    The maximum distance from the water water body that the animal should
                    be in.

            Returns
            -------
                None
        """
        # Store the datasets in the class variables.
        InteractiveRadar.__habitat_data = habitat
        InteractiveRadar.__traj_data = trajectories

        # First, create the date column on the trajectory dataset.
        InteractiveRadar.__traj_data = temp.create_date_column(InteractiveRadar.__traj_data)

        # Now, filter out the moving water bodies.
        a = habitat[(habitat['EcoGener'] == 'WR')]
        water_bodies = a.loc[(a['DistEWat'] == 0)]

        # Add an extra column that has bounding boxes for all the water bodies.
        bboxes = []
        for i in range(len(water_bodies)):
            lat = water_bodies.iloc[i]['lat']
            lon = water_bodies.iloc[i]['lon']
            bboxes.append(filt.get_bounding_box_by_radius(lat, lon, dist_from_water))

        water_bodies['bbox'] = bboxes

        # Name all the water bodies and store their respective data rows in a dictionary.
        point_dict = dict()
        for i in range(len(water_bodies)):
            point_dict[f'WaterBody #{i + 1}'] = water_bodies.iloc[i]

        InteractiveRadar.__point_dict = point_dict

        # Create the dropdown widget.
        InteractiveRadar.__list = widgets.Dropdown(options=list(point_dict.keys()),
                                                   value='WaterBody #1',
                                                   description='WaterBody: ',
                                                   disabled=False, continuous_update=False)

        # Show the plot.
        ie = widgets.interactive_output(InteractiveRadar.__plot_radar,
                                        {'body_name': InteractiveRadar.__list})

        # Display the widget and its output side-by-side.
        display(InteractiveRadar.__list, ie)

    @staticmethod
    def __plot_radar(body_name):
        # Get the water body data by its name.
        point = InteractiveRadar.__point_dict[body_name]

        # Filter the trajectory dataset by the water body.
        dataset = filt.filter_by_bounding_box(InteractiveRadar.__traj_data, point['bbox'])

        small_df = pd.DataFrame(columns=['traj_id', 'days', 'Species']).set_index('traj_id')
        ids_ = dataset.reset_index()['traj_id'].unique()
        for j in range(len(ids_)):
            mini = dataset.reset_index().loc[dataset.reset_index()['traj_id'] == ids_[j]]
            if 'D' in ids_[j]:
                small_df.loc[ids_[j]] = [pd.to_datetime(mini.Date).nunique(), 'Deer']
            elif 'E' in ids_[j]:
                small_df.loc[ids_[j]] = [pd.to_datetime(mini.Date).nunique(), 'Elk']
            else:
                small_df.loc[ids_[j]] = [pd.to_datetime(mini.Date).nunique(), 'Cattle']

        fig = px.scatter_polar(small_df, r='days', color='Species',
                               width=7 * 96, height=5 * 96,
                               title=f'Number of Days Spent Around {body_name}',
                               template='plotly_dark')

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True),
            ),
        )

        fig.update_polars(
            angularaxis_tickmode='array',
            angularaxis_tickvals=[0, 90, 180, 270],
            angularaxis_ticktext=['', '', '', ''],
        )

        fig.add_annotation(
            dict(
                font=dict(
                    color='white', size=12),
                x=0.03,
                y=0.5,
                showarrow=True,
                text=f"{body_name} Description<br><br>"
                     f"Name: {point['CowPast']}<br>"
                     f"Total Animals: {len(small_df)} <br>"
                     f"Coordinates: {round(point['lat'], 2), round(point['lon'], 2)} <br>"
                     f"Elevation: {point['Elev']} m <br>"
                     f"Canopy Cover: {point['Canopy']}%",
                textangle=0,
                align='left'
            )
        )

        fig.show()
