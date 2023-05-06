# adapt from https://github.com/uestc-db/traj-compression/blob/master/batch/TD-TR/TD-TR.cpp

import numpy as np
import time

import pandas as pd


def calc_SED(pA, pI, pB):
    """
    It computes the Synchronous Euclidean Distance (SED) error

    :param pA: initial point
    :param pI: middle point
    :param pB: final point
    :return: SED error
    """
    pA_lat, pA_lon, pA_time = pA
    pI_lat, pI_lon, pI_time = pI
    pB_lat, pB_lon, pB_time = pB

    middle_dist = pI_time - pA_time
    total_dist = pB_time - pA_time
    if total_dist == 0:
        time_ratio = 0
    else:
        time_ratio = middle_dist / total_dist

    lat = pA_lat + (pB_lat - pA_lat) * time_ratio
    lon = pA_lon + (pB_lon - pA_lon) * time_ratio

    lat_diff = lat - pI_lat
    lon_diff = lon - pI_lon
    error = np.sqrt((lat_diff * lat_diff) + (lon_diff * lon_diff))
    return error


def calc_DP(pA, pI, pB):
    """
    It computes the Perpendicular Distance (PD)

    :param pA: initial point
    :param pI: middle point
    :param pB: final point
    :return: shortest distance
    """
    pA_lat, pA_lon, pA_time = pA
    pI_lat, pI_lon, pI_time = pI
    pB_lat, pB_lon, pB_time = pB

    # equation: (yA−yB)x−(xA−xB)y+xAyB−xByA=0.
    A = pA_lon - pB_lon
    B = - (pA_lat - pB_lat)
    C = pA_lat * pB_lon - pB_lat * pA_lon

    if A == 0 and B == 0:
        shortDist = 0
    else:
        shortDist = abs((A * pI_lat + B * pI_lon + C) / np.sqrt(A * A + B * B))

    return shortDist


def calc_AVS(pA, pI, pB):
    """
    It computes the absolute value of speed (AVS)

    :param pA: initial point
    :param pI: middle point
    :param pB: final point
    :return: AVS value
    """
    pA_lat, pA_lon, pA_time = pA
    pI_lat, pI_lon, pI_time = pI
    pB_lat, pB_lon, pB_time = pB

    d1 = np.sqrt((pI_lat - pA_lat) * (pI_lat - pA_lat) + (pI_lon - pA_lon) * (pI_lon - pA_lon))
    d2 = np.sqrt((pB_lat - pI_lat) * (pB_lat - pI_lat) + (pB_lon - pI_lon) * (pB_lon - pI_lon))

    v1 = 0
    v2 = 0
    if (pI_time - pA_time) > 0:
        v1 = d1 / (pI_time - pA_time)
    if (pB_time - pI_time) > 0:
        v2 = d2 / (pB_time - pI_time)
    AVS = abs(v2 - v1)

    return AVS


def calc_TR_SP(trajectory, dim_set, traj_time, epsilon, epsilon2, calc_func, calc_func2):
    """
    It compress the trajectory using the two compression techniques.
    It is a recursive method, dividing the trajectory and compression both parts.

    :param trajectory: a single trajectory or a part of if
    :param dim_set: the attributes in the dict trajectory
    :param traj_time: the array with the time in seconds of each point
    :param epsilon: the threshold for the first compression
    :param epsilon2: the threshold for second compression
    :param calc_func: the measure for the first selected compression
    :param calc_func2: the measure for the second selected compression
    :return: the compressed trajectory (dict)
    """
    new_trajectory = {}
    for dim in dim_set:
        new_trajectory[dim] = np.array([])
    traj_len = len(trajectory['lat'])

    # time in seconds
    dmax, idx, _ = traj_max_dists(trajectory, traj_time, calc_func)
    start_location = (trajectory['lat'][0], trajectory['lon'][0], traj_time[0])
    final_location = (trajectory['lat'][-1], trajectory['lon'][-1], traj_time[-1])
    middle_location = (trajectory['lat'][idx], trajectory['lon'][idx], traj_time[idx])
    d_idx = calc_func2(start_location, middle_location, final_location)

    trajectory['time'] = trajectory['time'].astype(str)

    if (dmax > epsilon) & (d_idx > epsilon2):
        traj1 = {}
        traj2 = {}
        for dim in dim_set:
            traj1[dim] = trajectory[dim][0:idx]
            traj2[dim] = trajectory[dim][idx:]

        # compression of the parts
        recResults1 = traj1
        if len(traj1['lat']) > 2:
            recResults1 = traj_compression(traj1, dim_set, traj_time[0:idx], calc_func, epsilon)

        recResults2 = traj2
        if len(traj2['lat']) > 2:
            recResults2 = traj_compression(traj2, dim_set, traj_time[idx:], calc_func, epsilon)

        for dim in dim_set:
            new_trajectory[dim] = np.append(new_trajectory[dim], recResults1[dim])
            new_trajectory[dim] = np.append(new_trajectory[dim], recResults2[dim])

    else:
        trajectory['time'] = trajectory['time'].astype(str)
        for dim in dim_set:
            new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][0])
            if traj_len > 1:
                new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][-1])

    return new_trajectory


def traj_max_dists(trajectory, traj_time, calc_func):
    """
    It computes the selected measure for all points in between.

    :param trajectory: a single dict trajectory having the keys as each attribute
    :param traj_time: an array with the seconds of each point
    :param calc_func: the measure for the selected compression
    :return: the maximum distance, the index that provide the maximum distance, and the average of distances
    """
    dmax = 0
    idx = 0
    ds = np.array([])
    traj_len = len(trajectory['lat'])
    # start and final points
    start_location = (trajectory['lat'][0], trajectory['lon'][0], traj_time[0])
    final_location = (trajectory['lat'][-1], trajectory['lon'][-1], traj_time[-1])
    for i in range(1, (traj_len - 1)):
        # middle point at index i
        middle = (trajectory['lat'][i], trajectory['lon'][i], traj_time[i])
        # compute the distance
        d = calc_func(start_location, middle, final_location)
        # get distances information
        ds = np.append(ds, d)
        if d > dmax:
            dmax = d
            idx = i

    return dmax, idx, ds.mean()


def traj_compression(trajectory, dim_set, traj_time, calc_func, epsilon):
    """
    It compress the trajectory using the compression technique.
    It is a recursive method, dividing the trajectory and compression both parts.

    :param trajectory: a single trajectory or a part of if
    :param dim_set: the attributes in the dict trajectory
    :param traj_time: the array with the time in seconds of each point
    :param calc_func: the measure for the selected compression
    :param epsilon: the threshold
    :return: the compressed trajectory (dict)
    """
    new_trajectory = {}
    for dim in dim_set:
        new_trajectory[dim] = np.array([])
    traj_len = len(trajectory['lat'])

    # time in seconds
    dmax, idx, _ = traj_max_dists(trajectory, traj_time, calc_func)
    trajectory['time'] = trajectory['time'].astype(str)

    # print(f'\tepsilon: {epsilon}, dmax: {dmax}, index: {idx}, trajlen: {traj_len}')
    if dmax > epsilon:
        traj1 = {}
        traj2 = {}
        for dim in dim_set:
            traj1[dim] = trajectory[dim][0:idx]
            traj2[dim] = trajectory[dim][idx:]

        # compression of the parts
        recResults1 = traj1
        if len(traj1['lat']) > 2:
            recResults1 = traj_compression(traj1, dim_set, traj_time[0:idx], calc_func, epsilon)

        recResults2 = traj2
        if len(traj2['lat']) > 2:
            recResults2 = traj_compression(traj2, dim_set, traj_time[idx:], calc_func, epsilon)

        for dim in dim_set:
            new_trajectory[dim] = np.append(new_trajectory[dim], recResults1[dim])
            new_trajectory[dim] = np.append(new_trajectory[dim], recResults2[dim])

    else:
        trajectory['time'] = trajectory['time'].astype(str)
        for dim in dim_set:
            new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][0])
            if traj_len > 1:
                new_trajectory[dim] = np.append(new_trajectory[dim], trajectory[dim][-1])

    return new_trajectory


def compression(dataset, metric='TR', verbose=True, alpha=1):
    """
    It compress the dataset of trajectories using the selected compression technique.

    :param dataset: dict dataset with trajectories
    :param metric: the compression technique or combination of them (Default: TR).
    :param verbose: if True, it shows the messages (Default: True).
    :param alpha: the predefined factor (Default: 1).
    :return: the compressed dataset, compression rate, and processing time.
    """
    # sys.setrecursionlimit(2200)
    metrics = {'TR': calc_SED,
               'DP': calc_DP,
               'SP': calc_AVS,
               'TR_SP': calc_TR_SP,
               'SP_TR': calc_TR_SP,
               'SP_DP': calc_TR_SP,
               'DP_SP': calc_TR_SP,
               'DP_TR': calc_TR_SP,
               'TR_DP': calc_TR_SP}

    calc_func = metrics[metric]

    mmsis = list(dataset.keys())
    new_dataset = {}
    compression_rate = np.array([])
    processing_time = np.array([])

    dim_set = dataset[mmsis[0]].keys()
    print("Dim set is: ", dim_set)

    if verbose:
        print(f"Compressing with {metric} and factor {alpha}")
    for id_mmsi in range(len(mmsis)):
        new_dataset[mmsis[id_mmsi]] = {}
        if verbose:
            print(f"\tCompressing {id_mmsi} of {len(mmsis)}")
        # trajectory a
        t0 = time.time()
        curr_traj = dataset[mmsis[id_mmsi]]
        # get time in seconds
        traj_time = curr_traj['time'].astype('datetime64[s]')
        traj_time = np.hstack((0, np.diff(traj_time).cumsum().astype('float')))
        traj_time = traj_time / traj_time.max()
        # compress trajectory
        compress_traj = curr_traj
        try:
            if metric in ['TR_SP']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_SED)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_AVS)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_SED,
                                          calc_AVS)
            elif metric in ['SP_TR']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_AVS)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_SED)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_AVS,
                                          calc_SED)
            elif metric in ['SP_DP']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_AVS)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_DP)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_AVS,
                                          calc_DP)
            elif metric in ['TR_DP']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_SED)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_DP)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_SED,
                                          calc_DP)
            elif metric in ['DP_SP']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_DP)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_AVS)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_DP,
                                          calc_AVS)
            elif metric in ['DP_TR']:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_DP)
                max_epsilon2, idx2, epsilon2 = traj_max_dists(curr_traj, traj_time, calc_SED)
                compress_traj = calc_func(curr_traj, dim_set, traj_time, epsilon * alpha, epsilon2 * alpha, calc_DP,
                                          calc_SED)
            else:
                max_epsilon, idx, epsilon = traj_max_dists(curr_traj, traj_time, calc_func)
                compress_traj = traj_compression(curr_traj, dim_set, traj_time, calc_func, epsilon * alpha)
        except:
            print(
                f"\t\tIt was not possible to compress this trajectory {mmsis[id_mmsi]} of length {len(curr_traj['lat'])}.")

        compress_traj['time'] = compress_traj['time'].astype('datetime64[s]')
        new_dataset[mmsis[id_mmsi]] = compress_traj
        t1 = time.time() - t0
        if verbose:
            print(f"\tlength before: {len(curr_traj['lat'])}, length now: {len(compress_traj['lat'])}, reduction of {1 - len(compress_traj['lat'])/len(curr_traj['lat'])}")
        compression_rate = np.append(compression_rate, 1 - (len(compress_traj['lat']) / len(curr_traj['lat'])))
        processing_time = np.append(processing_time, t1)

    return new_dataset, compression_rate, processing_time


if __name__ == '__main__':
    dataset = pd.read_csv('../../examples/data/compression_data.csv')
    new_dataset = {}
    ids = dataset['mmsi'].unique()

    for id in ids:
        # getting one trajectory
        trajectory = dataset[dataset['mmsi'] == id]
        trajectory.set_index(['mmsi'])

        # converting trajectory to dict
        new_dataset[id] = {}
        for col in trajectory.columns:
            new_dataset[id][col] = np.array(trajectory[col])

    # Compression
    new_dataset, compression_rate, processing_time = compression(new_dataset, verbose=True)
