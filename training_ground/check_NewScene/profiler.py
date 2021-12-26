# Library Import
import numpy as np
import os
import matplotlib.pyplot as plt

# Init. path
data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

# Load all the data frames
score = np.load(data_path + 'score_history.npy', allow_pickle=False)
avg = np.load(data_path + 'avg_history.npy', allow_pickle=False)
distance = np.load(data_path + 'distance_history.npy', allow_pickle=False)

# Generate graphs
plt.figure(1)
plt.plot(score, alpha=0.85, label='ACC. Rewards')
plt.plot(avg, label='AVG. Rewards')
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Training Profile')
plt.savefig(data_path + 'Training_Profile.png')

plt.figure(2)
plt.plot(distance, label='Final Distance Error')
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Euclidean Distance Erroe')
plt.legend(loc='best')
plt.title('Distance-Error Profile')
plt.savefig(data_path + 'Distance_Profile.png')
