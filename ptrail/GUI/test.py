import distutils

from ptrail.preprocessing.filters import Filters
import inspect

a = inspect.getfullargspec(Filters.filter_by_date)

print(a.args)

print()