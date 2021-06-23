import multiprocessing

from core.TrajectoryDF import NumPandasTraj
from utilities import constants as const


class Helpers:
    @staticmethod
    def date_extractor(data, id_: int, dictionary: list):
        """
            Based on the Trajectory ID given, extract the datetime of the Trajectory point
            and the just extract the date from that point.

            Parameters
            ----------
                data: pandas.core.dataframe.DataFrame
                    The dataframe containing all the trajectory data.
                id_: int
                    The trajectory id for which date is to be extracted.
                dictionary: dict
                    This is for appending results. This is used due to multiprocessing.
                sema: Semaphore
                    The semaphore to restrict the number of processes to number of cores.
            Returns
            -------
                numpy.array
                    The numpy array containing dates.
        """
        matches = data.loc[data[const.TRAJECTORY_ID] == id_, [const.DateTime]]
        dictionary[id_] = matches[const.DateTime].dt.date

    @staticmethod
    def create_date_column(dataframe: NumPandasTraj, inplace=False):
        """
            From the DateTime column already present in the data, extract only the date
            and then add another column containing just the date.

            Parameters
            ----------
                dataframe: NumPandasTraj
                    The DaskTrajectoryDF on which the creation of the date column is to be done.
                inplace: bool
                    Whether to apply changes to the given dataframe or just return a new pandas DF.

            Returns
            -------
                pandas.core.dataframe.DataFrame
                    The dataframe containing the date column.
                    Dask Pandas DF is returned if the value of inplace parameter is False.

        """
        # If inplace is true, then continue with the original dataframe.
        # Otherwise, make a copy of the dataframe and then return it.
        if inplace:
            data = dataframe.reset_index(drop=False)
        else:
            data = dataframe.copy().reset_index(drop=False)

        ids = data.traj_id.unique().tolist()  # Grabbing all the unique trajectory ids.

        # Here, we are basically creating a multiprocessing manager and its dictionary for storing
        # the parallel return values. Apart from that, there is also a process_list which stores
        # the processes created which are to be run in parallel. Note that we dont have to worry here
        # about deadlocks and data corruption since all the processes running parallel are working
        # on different points of data.

        manager = multiprocessing.Manager()
        dictionary = manager.dict()
        process_list = []
        # Create as many processes as there are number of unique trajectory ids.
        for i in range(len(ids)):
            process = multiprocessing.Process(target=pH.date_extractor,
                                              args=(data, ids[i], dictionary))
            process_list.append(process)

        # Run and the processes parallel and then join all of them once they are done
        # with their execution.
        for j in process_list:
            j.start()

        for i in process_list:
            i.join()

        # for val in dictionary.values():
        #     dates.extend(val)
        # Since all the values given out by the parallel process are stored in a dictionary,
        # join all of them together and then assign an new column to the dataframe and
        # return the dataframe.
        # WARNING: DO NOT TO FORGET TO REASSIGN THE INDICES BACK TO DateTime AND traj_id.
        dates = []
        for val in dictionary.values():
            dates.extend(val)

        data['Date'] = dates

        data.set_index([const.DateTime, const.TRAJECTORY_ID], inplace=True, drop=True)
        return data
