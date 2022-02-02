import pandas as pd

from ptrail.preprocessing.filters import Filters
from ptrail.core.TrajectoryDF import PTRAILDataFrame

df = PTRAILDataFrame(data_set=pd.read_csv('/home/yjharanwala/Desktop/PTRAIL/examples/data/seagulls.csv'),
                     datetime='DateTime',
                     traj_id='traj_id',
                     latitude='lat',
                     longitude='lon')

print(df.columns)

df.drop(columns=['event-id'])

print(type(df))