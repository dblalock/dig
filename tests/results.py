#!/usr/bin/env python

import numpy as np
import pandas as pd

# TODO this file is hideous (but necessarily so for deadline purposes...)
#
# Also, this file is tightly coupled to figs.py; it basically has a func
# for each figure func that spits out data in exactly the required form


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


def main():
    print _extract_thruput('2.456 (1302931596/s), 2.344 (1365187713/s), 2.125 (1505882352/s), 2.829 (1131141746/s), 2.148 (1489757914/s), 2.167 (1476695892/s), 2.327 (1375161151/s), 2.145 (1491841491/s), 2.12 (1509433962/s), 2.112 (1515151515/s)')



if __name__ == '__main__':
    main()
