# Library Import
import numpy as np
import os
import matplotlib.pyplot as plt

# Init. path
path = os.getcwd()

# Load all the data frames
score = np.load(
    path + '/robotics-lab-project/proof_of_concept/\
        reach/data/score_history.npy',
    allow_pickle=False)
avg = np.load(
    path + '/robotics-lab-project/proof_of_concept/\
        reach/data/avg_history.npy',
    allow_pickle=False)

# Generate graphs
plt.plot(score, alpha=0.85, label='ACC. Rewards')
plt.plot(avg, label='AVG. Rewards')
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Training Profile')
plt.savefig(
    path + '/robotics-lab-project/proof_of_concept/\
        reach/data/' + 'Training Profile.png')
