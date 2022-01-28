# Library Import
import numpy as np
import os
import matplotlib.pyplot as plt

# Init. path
data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

# Load all the data frames
score = np.load(data_path + 'score_history.npy', allow_pickle=False)
avg = np.load(data_path + 'avg_history.npy', allow_pickle=False)

# Generate graphs
plt.figure(1)
plt.plot(score, alpha=0.5, label='Episodic Sum')
plt.plot(avg, label='Moving Average')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Testing Profile')
plt.savefig(data_path + 'Testing_Profile.png')
