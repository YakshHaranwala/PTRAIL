from ptrail.features.kinematic_features import KinematicFeatures
import inspect

a = inspect.getfullargspec(KinematicFeatures.generate_kinematic_features)

print(a)

print(type(a.args.remove('dataframe')))
print(a.args)