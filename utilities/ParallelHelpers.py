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