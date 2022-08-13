# Report up to epoch 100, get max test acc.
import os, itertools, sys
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

resDict = {}
accDict = {}
folder = sys.argv[1]

q = os.listdir(folder)
for item in os.listdir(folder):
    casename = item
    path = f'{folder}/{casename}'
    temp = []
    max_accs = []
    for dir in os.listdir(path):
        if dir == 'no':
            continue

        if not os.path.exists(f'{path}/{dir}/res'):
            continue

        max_acc = 0
        with open(f'{path}/{dir}/det', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first = 1
            cnter = 0
            for row in csv_reader:
                cnter = cnter + 1
                if cnter == 101:
                    break
                if first == 1:
                    first = 0
                else:
                    if max_acc < float(row[2]):
                        max_acc = float(row[2])
        # print(f'max is {max_acc}')
        max_accs.append(max_acc)

        with open(f'{path}/{dir}/res', 'r') as file:
            nums = [float(x.replace('\n', '')) for x in file.readlines()]
            temp.append(nums)

    # print(f'!@#!@# max_accs and temp size = {np.shape(max_accs)}---{np.shape(temp)}')
    temp = np.asarray(temp)
    temp = np.mean(temp, 0)
    if type(temp) == np.float64:
        if math.isnan(temp):
            continue
    resDict[casename] = temp  # (temp,np.size(temp,0))
    accDict[casename] = np.mean(max_accs)
with open('out', 'w') as file:
    for k, v in sorted(resDict.items()):
        file.write(f'{k}-{v}|{accDict[k]}\n')


