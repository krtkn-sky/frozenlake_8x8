import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    accuracy_over_time = []  # List to store accuracy over episodes

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    episode_range = range(1, episodes + 1)

    def update_plot(frame):
        axs[0].cla()
        axs[1].cla()

        sum_rewards = np.zeros(frame)
        for t in range(frame):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

        axs[0].plot(episode_range[:frame], sum_rewards)
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')

        accuracy_over_time_frame = accuracy_over_time[:frame]
        axs[1].plot(episode_range[:frame], accuracy_over_time_frame)
        axs[1].set_title('Accuracy Over Time')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Accuracy')

    ani = FuncAnimation(fig, update_plot, frames=episodes, repeat=False, blit=False)

    for i in range(episodes):
        state = env.reset()  # Remove [0] subscript
        terminated = False      # True when fall in hole or reached goal
        correct_moves = 0

        while(not terminated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, info = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

            if reward == 1:
                correct_moves += 1

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1
        
        # Calculate accuracy for this episode
        accuracy = correct_moves / (i + 1)
        accuracy_over_time.append(accuracy)

        # Update the graph in real-time
        ani.event_source.stop()
        update_plot(i + 1)
        plt.pause(0.01)
        ani.event_source.start()

    env.close()

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

    # Save the final graph
    plt.savefig('frozen_lake8x8.png')
    plt.show()

if __name__ == '__main__':
    run(1000, is_training=True, render=True)
