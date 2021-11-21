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
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

from ptrail.core.TrajectoryDF import PTRAILDataFrame


class InteractiveDonut:
    @staticmethod
    def animal_by_pasture(trajectories: PTRAILDataFrame, habitat: pd.DataFrame):
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
        pass

    @staticmethod
    def area_donut(habitat: pd.DataFrame):
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

        # Create a circle at the center of the plot
        my_circle = plt.Circle((0, 0), 0.65, color='white')

        # Add text to the centre of the plot.
        my_text = plt.Text(-0.4, -0.15, f'    Breakdown of\n'
                                        'Starkey Forest Area\n'
                                        '     by Pastures')

        # Custom wedges
        plt.pie(x=areas['area'], labels=areas['pasture'],
                wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
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
