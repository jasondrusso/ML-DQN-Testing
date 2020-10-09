import gym
import numpy as np


nchain_env = gym.make('NChain-v0')


def naive_sum_reward_agent(env, num_episodes=500):
    # This table holds summed action/reward pairs
    r_table = np.zeros((5, 2))

    for g in range(num_episodes):
        s = env.reset()
        done = False

        while not done:
            if np.sum(r_table[s, :]) == 0:
                # Select a random action if table row is empty
                a = np.random.randint(0, 2)
            else:
                # Select action with highest cumulative reward
                a = np.argmax(r_table[s, :])

            new_s, r, done, _ = env.step(a)
            r_table[s, a] += r
            s = new_s

    return r_table


def q_learning_with_table(env, num_episodes=500):
    # Summed action/reward pairs
    q_table = np.zeros((5, 2))
    y = 0.95                        # y = Reward discount rate
    lr = 0.8                        # lr = Model learning rate

    for i in range(num_episodes):
        s = env.reset()
        done = False

        while not done:
            if np.sum(q_table[s, :]) == 0:
                # Make random selection if no action was previously taken
                a = np.random.randint(0, 2)
            else:
                # Select action with largest q value
                a = np.argmax(q_table[s, :])

            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s

    return q_table


def eps_greedy_q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((5, 2))
    y = 0.95                        # y = Reward discount rate
    eps = 0.5                       # epsilon = Threshold for random choice selection
    lr = 0.8                        # lr = Model learning rate
    decay_factor = 0.999            # decay_factor = Reduces random choice threshold by this amount each episode

    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False

        while not done:
            if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s, :])

            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s

    return q_table


def run_game(table, env):
    s = env.reset()
    tot_reward = 0
    done = False

    while not done:
        a = np.argmax(table[s, :])
        s, r, done, _ = env.step(a)
        tot_reward += r

    return tot_reward


def test_methods(env, num_iterations=100):
    winner = np.zeros((3,))

    for g in range(num_iterations):
        m0_table = naive_sum_reward_agent(env, 500)
        m1_table = q_learning_with_table(env, 500)
        m2_table = eps_greedy_q_learning_with_table(env, 500)

        m0 = run_game(m0_table, env)
        m1 = run_game(m1_table, env)
        m2 = run_game(m2_table, env)

        w = np.argmax(np.array([m0, m1, m2]))
        winner[w] += 1
        print('Game {} of {}'.format(g + 1, num_iterations))

    return winner


reward_table = naive_sum_reward_agent(nchain_env)
print('\nRewards from naive learning:\n{}'.format(reward_table))

reward_table = q_learning_with_table(nchain_env)
print('\nRewards from q learning:\n{}'.format(reward_table))

reward_table = eps_greedy_q_learning_with_table(nchain_env)
print('\nRewards from epsilon greedy q learning:\n{}\n'.format(reward_table))

print('Let\'s see which is the winning algorithm...\n')
winning_algo = test_methods(nchain_env)
print('\nThe winning algorithm is {}'.format(winning_algo))