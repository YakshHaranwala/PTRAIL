"""
    This File contains the visualization that is a Donut chart depicting the area of
    each pasture is sq. km. It is an interactive visualization since upon clicking each
    ring of the donut, a dot plot showing the distribution of animals is also shown.

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
import geopandas as gpd
from IPython.core.display import display
from ipywidgets import widgets
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.preprocessing.filters import Filters as filt


class InteractiveDonut:
    # Class variables to store datasets and the widget.
    __habitat_data = None
    __traj_data = None
    __dropdown = None

    @staticmethod
    def animals_by_pasture(trajectories: PTRAILDataFrame, habitat: pd.DataFrame):
        """
            Plot a donut chart that shows the proportion of animals for each pasture.

            Parameters
            ----------
                trajectories: PTRAILDataFrame
                    The dataframe that contains trajectory data.
                habitat: pd.DataFrame
                    The dataframe that contains habitat data.

            Returns
            -------
                None
        """
        # Store the datasets in the class variables.
        InteractiveDonut.__traj_data = trajectories
        InteractiveDonut.__habitat_data = gpd.GeoDataFrame(habitat.reset_index(),
                                                           geometry=gpd.points_from_xy(habitat['lon'], habitat['lat']))

        # The list of available pastures.
        habitats = ['MDWCRK', 'SMITH-BALLY', 'STRIP', 'HORSE', 'BEAR', 'HALFMOON']

        # Create the dropdown widget.
        InteractiveDonut.__dropdown = widgets.Dropdown(options=habitats,
                                                       value='SMITH-BALLY',
                                                       description='Pasture',
                                                       disabled=False)

        # Create the widgets.
        ie = widgets.interactive_output(InteractiveDonut.__plot_pasture_donut,
                                        {'pasture_name': InteractiveDonut.__dropdown})

        # Display the widget and its output side-by-side.
        display(ie, InteractiveDonut.__dropdown)

    @staticmethod
    def __plot_pasture_donut(pasture_name):
        """
            Plot the donut chart as per the pasture.

            Parameters
            ----------
                pasture_name: str
                    The name of the pasture for which the Donut chart is to be plotted.

            Returns
            -------
                None
        """
        small = InteractiveDonut.__habitat_data.loc[InteractiveDonut.__habitat_data['CowPast'] == pasture_name]
        animals = InteractiveDonut._get_count_by_pasture(small,
                                                         InteractiveDonut.__traj_data)

        deer, cattle, elk = 0, 0, 0
        for i in range(len(animals)):
            if 'D' in animals[i]:
                deer += 1
            elif 'E' in animals[i]:
                elk += 1
            else:
                cattle += 1

        # Convert the data created above to a dict.
        data = {'Deer': deer, 'Elk': elk, 'Cattle': cattle}

        # Sort the above dictionary by values.
        data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1])}

        # Create a circle at the center of the plot
        my_circle = plt.Circle((0, 0), 0.65, color='white')

        # Add text to the centre of the plot.
        my_text = plt.Text(-0.3, -0.15, f'Breakdown of\n'
                                        f'  animals in\n' +
                                        pasture_name)

        # Custom wedges
        plt.pie(x=data.values(), labels=data.keys(),
                wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
                startangle=90)
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        p.gca().add_artist(my_text)

        plt.tight_layout()

    @staticmethod
    def _get_count_by_pasture(habitat: gpd.GeoDataFrame, trajectories: PTRAILDataFrame):
        """
            Filter the dataset by pasture and return the counts of deer, elk
            and cattle individually.

            Parameters
            ----------
                habitat: gpd.GeoDataFrame
                    The dataframe containing the habitat data.
                trajectories: PTRAILDataFrame
                    The dataframe containing the Trajectory data.

            Returns
            -------
                dict:
                    animal, count pairs.
        """
        # Using GeoPandas, get the bounding box of the pasture.
        bbox = habitat.geometry.total_bounds

        # Since GeoPandas uses lon, lat format, we need to swap it
        # in order to make it lat, lon format.
        bbox[0], bbox[1] = bbox[1], bbox[0]
        bbox[2], bbox[3] = bbox[3], bbox[2]
        bbox = tuple(bbox)

        # Using PTRAIL, filter the points that are inside the bounding box.
        filtered_df = filt.filter_by_bounding_box(dataframe=trajectories, bounding_box=bbox, inside=True)
        filtered_df = filtered_df.reset_index()

        return filtered_df.reset_index()['traj_id'].unique().tolist()

    @staticmethod
    def plot_area_donut(habitat: pd.DataFrame):
        """
            Given the trajectories and the habitat dataset, plot a donut plot
            which shows the area of each individual pasture as a ring and then
            has an interactive element that shows the distribution of animals
            upon clicking the pasture ring.

            Parameters
            ----------
                habitat: pd.core.dataframe.DataFrame
                    The dataset containing the habitat data.

            Returns
            -------
                None
        """
        # Get the area data for the habitat.
        areas = InteractiveDonut._get_pasture_area(habitat)
        areas = areas.sort_values(by='area')

        # Create a circle at the center of the plot
        my_circle = plt.Circle((0, 0), 0.65, color='white')

        # Add text to the centre of the plot.
        my_text = plt.Text(-0.4, -0.15, f'    Breakdown of\n'
                                        'Starkey Forest Area\n'
                                        '     by Pastures')

        # Custom wedges
        plt.pie(x=areas['area'], labels=areas['pasture'],
                wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
                startangle=90)
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        p.gca().add_artist(my_text)

        plt.tight_layout()

    @staticmethod
    def _get_pasture_area(dataset: pd.DataFrame):
        """
            Given the dataset containing the habitat data, return a dataframe
            containing the name of the pasture and the area of the pasture.

            Note
            ----
                It was noted that the starkey dataset had a pasture that did not have
                any name, hence it was renamed to STARK and if a pasture has less than
                3 points of data, then it is dropped since it does not have enough data
                to calculate its area. Moreover, if a pasture has an area of 0, it is
                dropped furthermore to clean the data.

            Note
            ----
                The area calculated is in km^2.

            Parameters
            ----------
                dataset: PTRAILDataFrame
                    The dataframe containing habitat data.

            Returns
            -------
                pd.core.dataframe.DataFrame:
                    The pandas dataframe containing the name of the pastures and their
                    respective areas.
        """
        # Convert the dataframe given into a GeoDataFrame.
        habitat_gdf = gpd.GeoDataFrame(dataset.reset_index(),
                                       geometry=gpd.points_from_xy(dataset['lon'], dataset['lat']))

        # Set the crs to EPSG:4326 and then make sure that it projected to EPSG:3857
        # since we want the area to be in metres.
        habitat_gdf.crs = "EPSG:4326"
        habitat_gdf = habitat_gdf.to_crs('EPSG:3857')

        # Get a list of all the unique habitats.
        habitats = habitat_gdf['CowPast'].unique()

        df = pd.DataFrame(columns=['pasture', 'area'])
        for val in habitats:
            # the try catch does the job of ignoring the pasture with less
            # than 2 points.
            try:
                small = habitat_gdf.loc[habitat_gdf['CowPast'] == val]

                # Rename the nan pasture to STARK
                if type(val) != str:
                    continue

                # Calculate the area and then append it to the dataframe.
                df.loc[val] = Polygon(small['geometry'].tolist()).area / 10e6
            except ValueError:
                continue

        # Clear the trash out of the DF and return it.
        df = df.reset_index().drop(columns=['pasture'])
        df = df.rename(columns={'index': 'pasture'})

        return df
