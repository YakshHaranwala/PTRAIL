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
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px

pd.options.mode.chained_assignment = None

sns.set()
class BarPlot:
    # Class variables to store datasets and the widget.
    __habitat_data = None
    __traj_data = None
    __point_dict = None
    __list = None

    @staticmethod
    def show_bar_plot(trajectories: PTRAILDataFrame, habitat: pd.DataFrame,
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
        BarPlot.__habitat_data = habitat
        BarPlot.__traj_data = trajectories

        # First, create the date column on the trajectory dataset.
        BarPlot.__traj_data = temp.create_date_column(BarPlot.__traj_data)
        BarPlot.__traj_data = temp.create_time_of_day_column(BarPlot.__traj_data)

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

        BarPlot.__point_dict = point_dict

        # Create the dropdown widget.
        BarPlot.__list = widgets.Dropdown(options=list(point_dict.keys()),
                                                   value='WaterBody #1',
                                                   description='WaterBody: ',
                                                   disabled=False, continuous_update=False)

        # Show the plot.
        ie = widgets.interactive_output(BarPlot.__plot_bar,
                                        {'body_name': BarPlot.__list})

        # Display the widget and its output side-by-side.
        display(BarPlot.__list, ie)

    @staticmethod
    def __plot_bar(body_name):
        # Get the water body data by its name.
        point = BarPlot.__point_dict[body_name]

        # Filter the trajectory dataset by the water body.
        dataset = filt.filter_by_bounding_box(BarPlot.__traj_data, point['bbox'])

        new_species = []
        for i in range(len(dataset)):
            if dataset['Species'].iloc[i] == 'D':
                new_species.append('Deer')
            elif dataset['Species'].iloc[i] == 'E':
                new_species.append('Elk')
            else:
                new_species.append('Cattle')

        dataset['Species'] = new_species

        small_df = dataset.groupby(by=['Time_Of_Day', 'Species']).count().reset_index()

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x='Time_Of_Day', y='lat', hue='Species', hue_order=['Deer', 'Elk', 'Cattle'], data=small_df, ax=ax)

        fig.text(1.1, 0.5,
                 f"{body_name} Description\n\n"
                 f"Name: {point['CowPast']}\n"
                 f"Coordinates: {round(point['lat'], 2), round(point['lon'],2)} \n"
                 f"Elevation: {point['Elev']} m \n"
                 f"Canopy Cover: {point['Canopy']}%"
                 )

        # fig.set_facecolor('black')
        ax.set_xlabel('TIME OF THE DAY')
        ax.set_ylabel('NUMBER OS TIMES VISITED')
        ax.set_title('Hydration trend of Species throughout the day')
        fig.tight_layout()


