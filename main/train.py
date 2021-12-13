from rai_gym.env import RAI_Env

if __name__ == '__main__':
    env = RAI_Env()
    env.reset()

    done = False
    while done is not True:
        action = env._random_action()
        reward, done = env.step(action)
        print(reward, done)
