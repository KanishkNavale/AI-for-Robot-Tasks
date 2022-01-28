# Robotics Lab Project 2021-22

This repository holds the project files of 'Practical Course Robotics: WS21-22'.

## Proof of Concept

1. OpenAI Gym Environments,

    * 'FetchReach-v1': The best agent is DDPG.
        |||
        |:--:|:--:|
        |<img src="https://github.com/KanishkNavale/Trajectory-Planning-using-HER-and-Reward-Engineering/blob/master/HER/data/test.gif?raw=true" width="300">| <img src="proof_of_concept/data/Training Profile.png" width="350">|

## Repository Setup Instructions

1. Clone & build [rai](https://github.com/MarcToussaint/rai) from the github following it's installation instructions.

2. Clone this repository.

    ```bash
    git clone https://github.com/KanishkNavale/robotics-lab-project
    ```

3. Add these in the .bashrc file

    ```bash
    # Misc. Alias
    alias python='python3'
    alias pip='pip3'

    # RAI Paths
    export PATH="$HOME/rai/bin:$PATH"
    export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/rai"

    # Practical Robotics Lab Project Package
    export PYTHONPATH="${PYTHONPATH}:$HOME/robotics-lab-project/"
    ```

4. Source the modified .bashrc file

    ```bash
    source ~/.bashrc
    ```

5. Install python package prequisites

    ```bash
    cd $HOME/robotics-lab-project
    pip install -r requirements.txt
    ```

## 1. Structure of the 'DDPG' Algorithm

<img src="https://www.researchgate.net/publication/342406026/figure/fig1/AS:906065046679559@1593034149792/Actor-critic-structure-for-DDPG-with-TSC.png" width="450">

## 2. Why Use Prioritized Experience Replay Buffer ?

|Without|With|
|:--:|:--:|
|<img src="presentation/pictures/without_per.png" width="350">| <img src="proof_of_concept/data/Training Profile.png" width="350">|

## 3. Training DDPG Agent for Point-to-Point Robot Trajectory

<img src="presentation/gifs/Reach_Training.gif" width="650">

|Training Profile|Testing Profile|
|:--:|:--:|
|<img src="training_ground/check_PyTorch/data/Training_Profile.png" width="300">| <img src="training_ground/check_PyTorch/data/Distance_Profile.png" width="300">|

## Developers

* Olga Klimashevska
* Kanishk Navale
