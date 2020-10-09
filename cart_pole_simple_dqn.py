from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import numpy as np
import gym
import matplotlib.pylab as plt
import multiprocessing


# Build simple sequential model with one flat hidden layer
model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 5)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Gym initialization
env = gym.make('NChain-v0')
num_episodes = 100

y = 0.95
eps = 0.5
decay_factor = 0.999
r_avg_list = []

for i in range(num_episodes):
    game_step = 0
    eps *= decay_factor
    r_sum = 0
    done = False

    s = env.reset()

    print("Episode {} of {}".format(i + 1, num_episodes))

    while not done:
        game_step += 1

        if np.random.random() < eps:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(model.predict(np.identity(5)[s:s + 1]))

        # print('Step {}: state = {}; selection = {}'.format(game_step, s, a))

        new_s, r, done, _ = env.step(a)
        target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
        target_vec = model.predict(np.identity(5)[s:s + 1])[0]

        target_vec[a] = target
        model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)

        s = new_s
        r_sum += r

    r_avg_list.append(r_sum / 1000)

plt.plot(r_avg_list)
plt.ylabel('Average reward per game')
plt.xlabel('Number of games')
plt.show()

for i in range(5):
    print('State {} - action {}'.format(i, model.predict(np.identity(5)[i:i + 1])))