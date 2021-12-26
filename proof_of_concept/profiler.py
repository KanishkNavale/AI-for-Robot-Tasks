# Library Import
import numpy as np
import os
import matplotlib.pyplot as plt

# Init. path
path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

# Load all the data frames
score = np.load(path + 'score_history.npy', allow_pickle=False)
avg = np.load(path + 'avg_history.npy', allow_pickle=False)

# Generate graphs
plt.plot(score, alpha=0.3, label='ACC. Rewards')
plt.plot(avg, label='AVG. Rewards')
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Training Profile')
plt.savefig(path + 'Training Profile.png')
