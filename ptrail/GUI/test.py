import distutils

from ptrail.preprocessing.filters import Filters
import inspect

a = inspect.getfullargspec(Filters.filter_by_date)

print(a.args)

print()

import seaborn as sns

a = sns.color_palette('colorblind').as_hex()

print(a)