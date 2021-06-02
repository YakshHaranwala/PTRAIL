"""
    This module contains all the helper functions for the parallel calculators in
    the spatial and temporal features classes.
    WARNING: These functions may not be used directly as they would result in a
             slower calculation and execution times. They are meant to be used
             only as helpers. For calculation of features, use the ones in the
             features package.
"""
import utilities.constants as const


class Helpers:
    @staticmethod
    def date_helper(dataframe):
        """
            This function is a helper method for the create_date_column(). The create_date_helper()
            methods delegates the actual task of creating the date to date_helper() function. What
            this function does is that it extracts the date from the DateTime column present in the
            DF and then adds a column to the DF itself, containing the date.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The DaskTrajectoryDF on which the creation of the date column is to be done.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the date column.

        """
        date_format = "%Y-%m-%d"  # Format of the date.

        # Reset the index of the DF in order to access the date time column and then generate
        # an iterable list of the items inside the column.
        gen = dataframe.reset_index()['DateTime'].iteritems()
        gen = list(gen)

        # Now, we extract the Date from all the time values.
        for i in range(len(gen)):
            gen[i] = gen[i][1].strftime(date_format)

        dataframe['Date'] = gen  # Assign the date column to the dataframe.
        return dataframe  # Return the dataframe with the date column inside it.

    @staticmethod
    def time_helper(dataframe):
        """
            This function is a helper method for the create_time_column(). The create_time_helper()
            methods delegates the actual task of creating the time to time_helper() function. What
            this function does is that it extracts the time from the DateTime column present in the
            DF and then adds a column to the DF itself, containing the time.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The DaskTrajectoryDF on which the creation of the time column is to be done.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the resultant time column.

        """
        time_format = "%H:%M:%S"

        # Reset the index of the DF in order to access the date time column and then generate
        # an iterable list of the items inside the column.
        datetime = dataframe.reset_index()['DateTime'].iteritems()
        datetime = list(datetime)

        # Now lets extract the time from the DateTime column.
        for i in range(len(datetime)):
            datetime[i] = datetime[i][1].strftime(time_format)

        dataframe['Time'] = datetime
        return dataframe

    @staticmethod
    def day_of_week_helper(dataframe):
        """
            This function is the helper function of the create_day_of_week() function. The day_of_week()
            function delegates the actual task of calculating the day of the week based on the datetime
            present in the data. This function does the calculation and creates a column called Day_Of_week
            and places it in the dataframe and returns it.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The dataframe on which calculation is to be performed.

            Returns
            -------
                pandas.core.dataframe
                    The dataframe containing the resultant Day_Of_Week column.
        """
        # First generate a list of the DateTime column of the dataframe using the iteritems()
        # function and then converting it into a python list.
        datetime = dataframe.reset_index()['DateTime'].iteritems()
        datetime = list(datetime)

        # Now extract all the names of the day based on the day.
        for i in range(len(datetime)):
            datetime[i] = datetime[i][1].day_name()

        # Assign the Day_Of_Week column and then return the dataframe.
        dataframe['Day_Of_Week'] = datetime

        return dataframe

    @staticmethod
    def weekend_helper(dataframe):
        """
             This function is the helper function of the create_weekend_indicator_week() function.
             The create_weekend_indicator() function delegates the actual task of checking whether
             the day of the week is either a Saturday or Sunday based on the datetime present in
             the data. This function does the calculation and creates a column called Weekend
             and places it in the dataframe and returns it.

             Parameters
             ----------
                 dataframe: NumPandasTraj
                     The dataframe on which calculation is to be performed.

             Returns
             -------
                 pandas.core.dataframe
                     The dataframe containing the resultant Day_Of_Week column.
        """
        # First, extract the DateTime column from the dataframe using the iteritems
        # and then convert it into a python list.
        weekend_indicator = dataframe.reset_index()['DateTime'].iteritems()
        weekend_indicator = list(weekend_indicator)

        # Now for each timestamp in the list, check its day and then append True/False
        # to the list based on whether the day is a weekday or weekend.
        for i in range(len(weekend_indicator)):
            weekend_indicator[i] = True if weekend_indicator[i][1].day_name() in const.WEEKEND else False

        # Append the column to the dataframe and return the DF.
        dataframe['Weekend'] = weekend_indicator
        return dataframe

    @staticmethod
    def time_of_day_helper(dataframe):
        """
             This function is the helper function of the create_time_of_day() function.
             The create_time_of_day() function delegates the actual task of calculating the time
             of the day of the week based on the datetime present in the data.This function does
             the calculation and creates a column called Time_Of_Day and places it in the dataframe
             and returns it.

             Parameters
             ----------
                 dataframe: NumPandasTraj
                     The dataframe on which calculation is to be performed.

             Returns
             -------
                 pandas.core.dataframe
                     The dataframe containing the resultant Day_Of_Week column.
        """
        # First, extract the DateTime column from the dataframe using the iteritems
        # and then convert it into a python list.
        timestamps = dataframe.reset_index()['DateTime'].iteritems()
        timestamps = list(timestamps)

        # Now, lets calculate the Time of the day based on the hour of time present
        # in the timestamp in the data and then append the results in a new column.
        for i in range(len(timestamps)):
            timestamps[i] = const.TIME_OF_DAY[timestamps[i][1].hour]

        # Now append the new column to the dataframe and return the dataframe.
        dataframe['Time_Of_Day'] = timestamps
        return dataframe
