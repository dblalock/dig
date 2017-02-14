#!/usr/bin/env python

import numpy as np
import pandas as pd

# TODO this file is hideous (but necessarily so for deadline purposes...)
#
# Also, this file is tightly coupled to figs.py; it basically has a func
# for each figure func that spits out data in exactly the required form
#
# Also-er, note that we're using the slowest Bolt impl ("u16 safe") which
# immediately upcasts LUT entries to 16 bits


def _extract_thruput(profile_str):
    rep_strs = profile_str.strip(',').split(',')
    # print rep_strs
    thruput_parens = [s.strip(' ').split(' ')[1] for s in rep_strs]
    # print thruput_parens
    return np.array([int(s.strip('()s/')) for s in thruput_parens])


def popcount_results():
    popcnt_times = {}
    popcnt_times[8] = '2.456 (1302931596/s), 2.344 (1365187713/s), 2.125 (1505882352/s), 2.829 (1131141746/s), 2.148 (1489757914/s), 2.167 (1476695892/s), 2.327 (1375161151/s), 2.145 (1491841491/s), 2.12 (1509433962/s), 2.112 (1515151515/s)'
    popcnt_times[16] = '4.368 (732600732/s), 4.121 (776510555/s), 3.926 (815078960/s), 4.105 (779537149/s), 4.176 (766283524/s), 4.119 (776887594/s), 4.464 (716845878/s), 4.153 (770527329/s), 4.364 (733272227/s), 4.198 (762267746/s)'
    popcnt_times[32] = '7.612 (420388859/s), 7.347 (435551925/s), 7.694 (415908500/s), 9.122 (350800263/s), 7.343 (435789186/s), 9.344 (342465753/s), 8.148 (392734413/s), 9.046 (353747512/s), 8.455 (378474275/s), 7.685 (416395575/s)'

    bolt_times = {}
    bolt_times[8] = '0.461 (2169197396/s), 0.456 (2192982456/s), 0.539 (1855287569/s), 0.53 (1886792452/s), 0.456 (2192982456/s), 0.452 (2212389380/s), 0.442 (2262443438/s), 0.438 (2283105022/s), 0.434 (2304147465/s), 0.547 (1828153564/s)'
    bolt_times[16] = '0.894 (1118568232/s), 1.08 (925925925/s), 0.88 (1136363636/s), 0.877 (1140250855/s), 0.881 (1135073779/s), 0.847 (1180637544/s), 1.011 (989119683/s), 0.866 (1154734411/s), 0.984 (1016260162/s), 0.838 (1193317422/s)'
    bolt_times[32] = '2.047 (488519785/s), 1.726 (579374275/s), 1.924 (519750519/s), 2.085 (479616306/s), 2.076 (481695568/s), 1.748 (572082379/s), 1.757 (569151963/s), 2.064 (484496124/s), 1.742 (574052812/s), 1.725 (579710144/s)'

    out_dicts = []
    algos = ['Bolt', 'Binary Embedding']
    dicts = [bolt_times, popcnt_times]
    for algo, d in zip(algos, dicts):
        for nbytes, s in d.items():
            bytes_str = "{}B".format(nbytes)
            thruputs = _extract_thruput(s)
            out_dicts += [{'algo': algo, 'nbytes': bytes_str, 't': t} for t in thruputs]

    return pd.DataFrame.from_records(out_dicts)


def query_speed_results():
    # NOTE: all thruputs in this function need be multiplied by 100,000
    # because we're reporting distances/sec, not time to query 100k points
    #   -EDIT: except actually, div by 10 cuz we plot it in millions

    bolt_times = {}
    bolt_times[8] = '4.385 (22805/s), 4.385 (22805/s), 4.408 (22686/s), 4.385 (22805/s), 5.117 (19542/s), 4.378 (22841/s), 4.392 (22768/s), 4.393 (22763/s), 4.381 (22825/s), 4.383 (22815/s)'
    bolt_times[16] = '8.268 (12094/s), 9.807 (10196/s), 8.389 (11920/s), 8.681 (11519/s), 8.711 (11479/s), 8.293 (12058/s), 9.797 (10207/s), 8.32 (12019/s), 9.767 (10238/s), 9.499 (10527/s)'
    bolt_times[32] = '19.385 (5158/s), 17.215 (5808/s), 18.612 (5372/s), 18.117 (5519/s), 17.323 (5772/s), 18.436 (5424/s), 18.979 (5268/s), 16.274 (6144/s), 19.696 (5077/s), 17.026 (5873/s)'

    pq_times = {}
    pq_times[8] = '36.499 (2739/s), 35.729 (2798/s), 36.521 (2738/s), 37.924 (2636/s), 37.079 (2696/s), 36.444 (2743/s), 36.115 (2768/s), 36.955 (2705/s), 35.913 (2784/s), 40.354 (2478/s)'
    pq_times[16] = '79.482 (1258/s), 82.546 (1211/s), 84.992 (1176/s), 84.996 (1176/s), 86.218 (1159/s), 84.495 (1183/s), 90.637 (1103/s), 82.164 (1217/s), 85.954 (1163/s), 82.255 (1215/s)'
    pq_times[32] = '214.85 (465/s), 217.41 (459/s), 212.49 (470/s), 210.75 (474/s), 211.12 (473/s), 212.54 (470/s), 209.91 (476/s), 219.95 (454/s), 212.97 (469/s), 213.44 (468/s)'

    opq_times = {}
    opq_times[8] = '38.653 (2587/s), 36.958 (2705/s), 37.684 (2653/s), 35.902 (2785/s), 38.032 (2629/s), 39.511 (2530/s), 42.321 (2362/s), 38.94 (2568/s), 39.224 (2549/s), 39.06 (2560/s)'
    opq_times[16] = '82.636 (1210/s), 82.401 (1213/s), 88.424 (1130/s), 86.649 (1154/s), 83.329 (1200/s), 82.719 (1208/s), 82.281 (1215/s), 80.581 (1240/s), 80.777 (1237/s), 81.107 (1232/s)'
    opq_times[32] = '221.61 (451/s), 230.01 (434/s), 241.68 (413/s), 222.39 (449/s), 215.13 (464/s), 215.49 (464/s), 212.27 (471/s), 213.95 (467/s), 213.96 (467/s), 217.79 (459/s)'

    # pairq times just call opq because query-time operations are identical
    pairq_times = {}
    pairq_times[8] = '37.999 (2631/s), 38.182 (2619/s), 37.472 (2668/s), 37.499 (2666/s), 36.748 (2721/s), 40.319 (2480/s), 39.961 (2502/s), 38.048 (2628/s), 39.389 (2538/s), 38.475 (2599/s)'
    pairq_times[16] = '84.2 (1187/s), 85.918 (1163/s), 84.121 (1188/s), 86.76 (1152/s), 83.399 (1199/s), 86.055 (1162/s), 88.24 (1133/s), 89.417 (1118/s), 87.597 (1141/s), 81.535 (1226/s)'
    pairq_times[32] = '209.43 (477/s), 203.78 (490/s), 210.62 (474/s), 218.42 (457/s), 202.41 (494/s), 214.8 (465/s), 200.74 (498/s), 200.19 (499/s), 205.14 (487/s), 201.71 (495/s)'

    # 1, 16 -> rowmajor times; 64, 256, 1024 -> colmajor times; (ie, use times from best layout)
    matmul1_times = '12.063 (8289811/s), 11.231 (8903926/s), 10.283 (9724788/s), 10.864 (9204712/s), 10.492 (9531071/s), 10.877 (9193711/s), 10.79 (9267840/s), 10.85 (9216589/s), 11.041 (9057150/s), 10.647 (9392317/s)'
    matmul16_times = '21.707 (73708941/s), 21.38 (74836295/s), 21.71 (73698756/s), 21.54 (74280408/s), 21.454 (74578167/s), 21.989 (72763654/s), 22.486 (71155385/s), 22.048 (72568940/s), 23.18 (69025021/s), 21.771 (73492260/s)'
    matmul64_times = '56.496 (113282356/s), 55.488 (115340253/s), 54.853 (116675478/s), 56.689 (112896681/s), 56.482 (113310435/s), 55.644 (115016893/s), 54.623 (117166761/s), 55.773 (114750865/s), 54.726 (116946241/s), 54.918 (116537383/s)'
    matmul256_times = '164.72 (155414306/s), 168.41 (152014488/s), 169.93 (150652927/s), 164.99 (155157157/s), 166.66 (153609831/s), 163.04 (157012830/s), 167.45 (152880544/s), 161.06 (158949936/s), 171.13 (149594750/s), 168.49 (151940505/s)'
    matmul1024_times = '653.63 (156664035/s), 677.26 (151197248/s), 692.88 (147788938/s), 664.79 (154032909/s), 702.61 (145742096/s), 651.74 (157116904/s), 656.4 (156003388/s), 664.69 (154056314/s), 665.34 (153906736/s), 651.88 (157083643/s)'

    out_dicts = []
    algos = ['Bolt', 'PQ', 'OPQ', 'PairQ']
    dicts = [bolt_times, pq_times, opq_times, pairq_times]

    for algo, d in zip(algos, dicts):
        for nbytes, s in d.items():
            # bytes_str = "{}B".format(nbytes)
            thruputs = _extract_thruput(s) / 10.
            # out_dicts += [{'algo': algo, 'nbytes': bytes_str, 'y': t} for t in thruputs]
            out_dicts += [{'algo': algo, 'nbytes': nbytes, 'y': t} for t in thruputs]

    matmul_strs = [matmul1_times, matmul16_times, matmul64_times, matmul256_times, matmul1024_times]
    batch_sizes = [1, 16, 64, 256, 1024]
    nbytes_list = [8, 16, 32]  # replicate results in each plot
    for s, sz in zip(matmul_strs, batch_sizes):
        algo = 'Matmul {}'.format(sz)
        for nbytes in nbytes_list:
            thruputs = _extract_thruput(s) / 1e6
            out_dicts += [{'algo': algo, 'nbytes': nbytes, 'y': t} for r in thruputs]

    return pd.DataFrame.from_records(out_dicts)


def main():
    print _extract_thruput('2.456 (1302931596/s), 2.344 (1365187713/s), 2.125 (1505882352/s), 2.829 (1131141746/s), 2.148 (1489757914/s), 2.167 (1476695892/s), 2.327 (1375161151/s), 2.145 (1491841491/s), 2.12 (1509433962/s), 2.112 (1515151515/s)')



if __name__ == '__main__':
    main()
