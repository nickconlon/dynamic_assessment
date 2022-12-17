import numpy as np

"""
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
https://stats.stackexchange.com/questions/34357/q-learning-in-a-stochastic-environment
"""


def train_q_learning_agent(env, alpha=0.1, gamma=0.6, epsilon=0.5, q_table=None, rounds=10000):
    env.stochastic_transitions = True
    if q_table is None:
        q_table = np.zeros([env.observation_space.n, env.action_space.n])
        #q_table = np.random.random(size=(env.observation_space.n, env.action_space.n))
        #q_table = np.random.normal(size=(env.observation_space.n, env.action_space.n))

    for i in range(1, rounds):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            print(f"Episode: {i}")

    print("Training finished.\n")
    return q_table


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import environment as my_env
    import configs as configs
    initial_state = configs.LOCATIONS[configs.AREA_2].position
    goal_location = configs.LOCATIONS[configs.HOME].position
    obstacles = []
    zones = []
    env = my_env.Environment(initial_state, goal_location, obstacles, zones)
    Q = train_q_learning_agent(env)

    for i in range(25):
        print('retrain {}/25'.format(i))
        x = int(np.floor(np.random.randint(0, 49)))
        y = int(np.floor(np.random.randint(0, 49)))
        env = my_env.Environment(np.asarray([x, y]), goal_location, obstacles, zones)
        Q = train_q_learning_agent(env, q_table=Q)
    np.save('q_{}.npy'.format(configs.HOME), Q)

