"""
We will need to learn how to grab data using osmnx regarding the POIs and stuff.
So far the features that I can think of:
    1. Find the points inside a polygon.
    2. Distance from a point to a water body.
    3. Distance from a point to nearby restaurants ( can give option of adding restaurant category
                                                     or prices to the function).
    4. Distance from a point to nearby tourist attractions.
    5. Most visited tourist attractions and the time when the most visitors are noted.
    6. Trajectories crossing paths (maybe near a POI or so many times each day something like that).
    7. Availability of Banks/ATMs from a point or from a restaurant.
    8. POIs intersected by a trajectory.
    9. Classify the behaviour of the trajectories into groups like commuters, tourists, meet behaviour
       etc. This is taken from the semantic_traj.pdf paper. For vehicle trajectories, we can also
       classify whether the vehicle is showing SpeedingBehaviour since it is very easy for us to
       calculate speeds. (pg. 8-10).
    10. Another interesting feature is to identify the leader object for animal herds and other
        animal trajectories. This can be done by identifying the Leader behaviour mentioned in the
        semantic_traj.pdf paper (pg. 9).
    11. Another very common one is the stop and stay point detection.
    12. Avoiding and chasing behaviour can also be done. However, it seems a little bit harder.
"""
