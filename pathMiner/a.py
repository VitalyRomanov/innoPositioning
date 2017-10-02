import numpy as np
import pandas as pd
from collections import namedtuple
import random
import sys
import math
from sklearn.metrics import classification_report
from scipy import sparse

import pandas as pd
import numpy as np
import matplotlib.pylab as plt


curr = []
arr = []


f = open('train.txt', 'r')
for line in f:
    if len(line) > 2:
        nums = [float(x) for x in line.strip().split(" ")]
        curr.append(nums)
    else:
        arr.append(curr)
        curr = []
data = np.array(arr)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def mean(p1, p2):
    a = [0.0]*2
    a[0] = (p1[0] + p2[0]) / 2.0
    a[1] = (p1[1] + p2[1]) / 2.0
    return a

def _dtw_distance(ts_a, ts_b, d = lambda x,y: abs(x-y)):
    
    # Create cost matrix via broadcasting with large int
    #ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxint * np.ones((M, N))
    
    k, l, minn = 0, 0 ,0
    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])
    
    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])
    
    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - 10),
                        min(N, i + 10)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

# Return DTW distance given window
    return cost[-1, -1]

def centrate(ts_a, ts_b, d = lambda x,y: abs(x-y)):
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxint * np.ones((M, N))
    temp_trajectory = []
    k, l, iii = 0, 0, 0
    minn = [0.0] * 2
    # Initialize the first row and column
    
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])
    
    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])
    
    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - 10),
                        min(N, i + 10)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
    temp_trajectory.append(mean(ts_a[0], ts_b[0]))
    while k + 1 < M or l + 1 < N:
        #print k + 1, l + 1, M - 2, N - 2, np.shape(cost)
        if k + 1 == M:
            minn[0] = k
            minn[1] = l + 1
        elif l + 1 == N:
            minn[0] = k + 1
            minn[1] = l
        else:
            b = [cost[k + 1][l + 1], cost[k][l + 1], cost[k + 1][l]]
            iii = b.index(min(b))
            if iii == 0:
                minn[0] = k + 1
                minn[1] = l + 1
            if iii == 1:
                minn[0] = k
                minn[1] = l + 1
            if iii == 2:
                minn[0] = k + 1
                minn[1] = l
                    
        temp_trajectory.append(mean(ts_a[minn[0]], ts_b[minn[1]]))
        k = minn[0]
        l = minn[1]
    i = 0
    new_trajectory = []
    new_trajectory.append(temp_trajectory[0])
    n = len(temp_trajectory)
    diff = distance(temp_trajectory[0], temp_trajectory[n - 1]) * 0.01
    while i < n - 1:
        if distance(temp_trajectory[i], temp_trajectory[i + 1]) > diff:
            new_trajectory.append(temp_trajectory[i + 1])
        i = i + 1
#print len(ts_a), len(ts_b), len(new_trajectory)
    return new_trajectory





def k_means_clust(data, num_clust, num_iter, w = 3):
    centroids = random.sample(data,num_clust)
    counter = 0
    new_data = []
    old_data = []
    ii = 0
    for n in range(num_iter):
        print(counter)
        counter += 1
        assignments = {}
        #assign data points to clusters
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                #print i, j
                cur_dist = _dtw_distance(i, j, distance)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []
        #recalculate centroids of clusters
        for key in assignments:
            for k in assignments[key]:
                old_data.append(data[k])
            while(len(old_data) > 1):
                while(ii + 1 < len(old_data)):
                    new_data.append(centrate(old_data[ii], old_data[ii + 1], distance))
                    ii = ii + 2
                old_data = new_data
                new_data = []
                ii = 0
            centroids[key] = old_data[0]
            old_data = []
                
                
    print(assignments)
    return centroids

centroids=k_means_clust(data,5,10,4)
for c_ind, i in enumerate(centroids):
    for ind, j in enumerate(i):
        plt.plot(i[ind])

plt.show()