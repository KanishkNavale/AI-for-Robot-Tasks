# Library Import
import numpy as np
import os
import matplotlib.pyplot as plt

# Init. path
data_path = os.getcwd() + '/main/check_aidedPosition/TD3/data/'

# Load all the data frames
score = np.load(data_path + 'score_history.npy', allow_pickle=False)
avg = np.load(data_path + 'avg_history.npy', allow_pickle=False)

# Clip First 100 episodes
end = len(score)
cut = 150
score = score[cut:]
avg = avg[cut:]

# Generate graphs
plt.plot(score, alpha=0.85, label='ACC. Rewards')
plt.plot(avg, label='AVG. Rewards')
plt.xlim([cut, end])
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Training Profile')
plt.savefig(data_path + 'Training Profile.png')
