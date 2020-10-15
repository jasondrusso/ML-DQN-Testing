import gym
import time

env = gym.make("SpaceInvaders-v0", obs_type='image')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.05)

env.close()