# Robotics Lab Project 2021-22

This repository holds the project files of 'Practical Course Robotics: WS21-22'.

## Proof of Concept

1. OpenAI Gym Environments,

    * 'FetchReach-v1': The best agent is DDPG.
        <p align="left">
            <img src="proof_of_concept/reach/data/Training Profile.png" width="350" title="hover text">
        </p>

## Repository Setup Instructions

1. Clone & build [rai-python](https://github.com/MarcToussaint/rai-python) from the github following it's installation instructions.

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
    export PATH="$HOME/rai-python/rai/bin:$PATH"
    export PYTHONPATH="${PYTHONPATH}:$HOME/rai-python/build"

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

## Developers

* Olga Klimashevska
* Kanishk Navale