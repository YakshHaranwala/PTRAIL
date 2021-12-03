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

    | Authors: Yaksh J Haranwala
"""
import pandas as pd
from IPython.core.display import display
from ipywidgets import widgets, AppLayout
from ptrail.core.TrajectoryDF import PTRAILDataFrame

from ptrail.features.temporal_features import TemporalFeatures as temp
from ptrail.preprocessing.filters import Filters as filt

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.options.mode.chained_assignment = None


class HydrationTrends:
    # Class variables to store datasets and the widget.
    __habitat_data = None
    __traj_data = None
    __point_dict = None
    __list = None

    @staticmethod
    def show_hydration_trends(trajectories: PTRAILDataFrame, habitat: pd.DataFrame,
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
        HydrationTrends.__habitat_data = habitat
        HydrationTrends.__traj_data = trajectories

        # First, create the date column on the trajectory dataset.
        HydrationTrends.__traj_data = temp.create_date_column(HydrationTrends.__traj_data)
        HydrationTrends.__traj_data = temp.create_time_of_day_column(HydrationTrends.__traj_data)

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

        HydrationTrends.__point_dict = point_dict

        # Create the dropdown widget.
        HydrationTrends.__list = widgets.Dropdown(options=list(point_dict.keys()),
                                                  value='WaterBody #1',
                                                  description='WaterBody: ',
                                                  disabled=False, continuous_update=False)

        # Show the plot.
        radar = widgets.interactive_output(HydrationTrends.__plot_radar,
                                           {'body_name': HydrationTrends.__list})

        bar = widgets.interactive_output(HydrationTrends.__plot_bar,
                                         {'body_name': HydrationTrends.__list})

        output = AppLayout(header=None, center=None, footer=None,
                           left_sidebar=bar, right_sidebar=radar,
                           pane_heights=[1, 1, '480px'])

        # Display the widget and its output side-by-side.
        display(HydrationTrends.__list, output)

    @staticmethod
    def __plot_radar(body_name):
        # Get the water body data by its name.
        point = HydrationTrends.__point_dict[body_name]

        # Filter the trajectory dataset by the water body.
        dataset = filt.filter_by_bounding_box(HydrationTrends.__traj_data, point['bbox'])

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

        fig1 = px.scatter_polar(small_df, r='days', color='Species',
                                title=f'Number of Days Spent Around {body_name}',
                                template="ggplot2", color_discrete_sequence=px.colors.qualitative.Vivid, )

        fig1.update_layout(
            polar=dict(
                radialaxis=dict(visible=True),
            ),
            title=dict(
                font={'size': 13}
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            autosize=False,
            width=540,
            height=280,
            paper_bgcolor="Gainsboro",
        )

        fig1.update_polars(
            angularaxis_tickmode='array',
            angularaxis_tickvals=[0, 90, 180, 270],
            angularaxis_ticktext=['', '', '', ''],
        )

        fig1.show()

    @staticmethod
    def __plot_bar(body_name):
        # Get the water body data by its name.
        point = HydrationTrends.__point_dict[body_name]

        # Filter the trajectory dataset by the water body.
        dataset = filt.filter_by_bounding_box(HydrationTrends.__traj_data, point['bbox'])

        new_species = []
        for i in range(len(dataset)):
            if dataset['Species'].iloc[i] == 'D':
                new_species.append('Deer')
            elif dataset['Species'].iloc[i] == 'E':
                new_species.append('Elk')
            else:
                new_species.append('Cattle')

        dataset['Species'] = new_species

        # Order the dataset according to the times of day.
        small_df = dataset.groupby(by=['Time_Of_Day', 'Species']).count().reset_index()
        times = ['Early Morning', 'Morning', 'Noon', 'Evening', 'Night', 'Late Night']
        small_df['Time_Of_Day'] = pd.Categorical(small_df['Time_Of_Day'],
                                                 categories=times,
                                                 ordered=True)
        small_df = small_df.sort_values(by='Time_Of_Day', ascending=True)

        # Plot the barplot from the data above.
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x='Time_Of_Day', y='lat', hue='Species', hue_order=['Deer', 'Elk', 'Cattle'], data=small_df, ax=ax)

        # Set the description of the text.
        fig.text(-0.4, 0.5,
                 f"{body_name} Description\n\n"
                 f"Pasture Name: {point['CowPast']}\n"
                 f"Coordinates: {round(point['lat'], 2), round(point['lon'], 2)} \n"
                 f"Elevation: {point['Elev']} m \n"
                 f"Canopy Cover: {point['Canopy']}%"
                 , fontdict={'size': 13})

        fig.set_facecolor('lightgray')
        ax.set_xlabel('TIME OF THE DAY')
        ax.set_ylabel('NUMBER OS TIMES VISITED')
        ax.set_title('Hydration trend of Species throughout the day')

        fig.tight_layout()
