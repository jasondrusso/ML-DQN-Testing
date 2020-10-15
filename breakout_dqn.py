import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Environment configuration and Q learning parameters
seed = 42
gamma = 0.99                                        # Discount for past rewards
epsilon = 1.0                                       # Epsilon greedy parameter (for random action selection)
epsilon_min = 0.1                                   # Minimum epsilon
epsilon_max = 1.0                                   # Maximum epsilon
epsilon_interval = (epsilon_max - epsilon_min)      # Rate to reduce chance of random action
batch_size = 32                                     # Size of replay buffer batch
max_steps_per_episode = 10000

# Replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []

# Counters
running_reward = 0
episode_count = 0
frame_count = 0

epsilon_random_frames = 50000           # Num frames for random observation
epsilon_greedy_frames = 1000000.0       # Num frames for action exploration
max_memory_length = 100000              # Max replay length (Use 100000 for 16GB Ram)
update_after_actions = 4                # Train model after 4 actions
update_target_network = 10000           # Steps to take before updating target network

loss_function = keras.losses.Huber()    # Using Huber loss for stability

# Define Q network
num_actions = 4

# Initialize Breakout environment using OpenAi Gym
env = gym.make("BreakoutNoFrameskip-v4")
env.seed(seed)


# Define Q network model
def create_q_model():
    inputs = layers.Input(shape=(210, 160, 3,))

    layer1 = layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation='relu')(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation='relu')(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation='relu')(layer4)
    action = layers.Dense(num_actions, activation='linear')(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for the Q-values which are used to select and action
model = create_q_model()

# The target model predicts future rewards. The weights of the target model get updated every
# 10000 steps. Thus when the loss between the Q-values is calculated, the target Q-value is stable.
model_target = create_q_model()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

while True:                             # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    for time_step in range(1, max_steps_per_episode):
        # Comment out if we don't desire to see games in action
        env.render()

        frame_count += 1

        # Use epsilon_greedy for action exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action q values from environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)

            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)
        # print('Next state looks like {}'.format(state_next.shape))

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)

        # Set up new state
        state = state_next
        episode_reward += reward

        # Train the model with sample data from replay buffer
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get random samples from replay buffer
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

            # Update Q values from sampling. Use target model for better stability.
            future_rewards = model_target.predict(state_next_sample)

            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss of updated Q value
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model
                q_values = model(state_sample)

                # Only update Q value for action taken using the mask obtained earlier
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

                # Calculate loss between old and new Q values
                loss = loss_function(updated_q_values, q_action)

            # Back propagation
            grads = tape.gradient(loss, model.trainable_variables)              # get learned gradients
            optimizer.apply_gradients(zip(grads, model.trainable_variables))    # multiply q variables by grads

        # Update target network
        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())
            template = 'Running reward: {:.2f} at episode {}, frame count {}'
            print(template.format(running_reward, episode_count, frame_count))

        # Pop the first entry from the rolling histories
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]                      # Pop 1st entry from history

    running_reward = np.mean(episode_reward_history)
    episode_count += 1

    if running_reward > 40:
        # Consider task solved
        print('Solved at episode {}!'.format(episode_count))
        break

model.save('models/breakout/model_stage_1')