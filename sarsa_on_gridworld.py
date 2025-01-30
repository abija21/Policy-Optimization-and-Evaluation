import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# True optimal value function
true_v = np.array([
    [2.6638, 2.9969, 2.8117, 3.6671, 4.8497],
    [2.9713, 3.5101, 4.0819, 4.8497, 7.1648],
    [2.5936, 0.0, 0.0, 0.0, 8.4687],
    [2.0993, 1.085, 0.0, 8.6097, 9.5269],
    [1.085, 4.9466, 8.4687, 9.5269, 0.0]
])

# Parameters for SARSA
alpha = 0.1
epsilon = 0.2
gamma = 0.925
episodes = 1000
runs = 20

# Environment setup
grid_size = (5, 5)
food_state = (4, 4)
monster_states = [(0, 3), (4, 1)]
forbidden_furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]
actions = ['AU', 'AD', 'AL', 'AR']
action_map = {'AU': (-1, 0), 'AD': (1, 0), 'AL': (0, -1), 'AR': (0, 1)}
action_index = {a: i for i, a in enumerate(actions)}


# Transition function
def next_state_modified(current, action):
    intended_move = action_map[action]
    noisy_moves = {
        'AU': ['AU', 'AL', 'AR', None],
        'AD': ['AD', 'AR', 'AL', None],
        'AL': ['AL', 'AU', 'AD', None],
        'AR': ['AR', 'AD', 'AU', None],
    }
    probs = [0.7, 0.12, 0.12, 0.06]
    chosen_action = np.random.choice(noisy_moves[action], p=probs)
    if chosen_action is None:
        return current
    move = action_map[chosen_action]
    next_state = (current[0] + move[0], current[1] + move[1])
    if (
            0 <= next_state[0] < grid_size[0]
            and 0 <= next_state[1] < grid_size[1]
            and next_state not in forbidden_furniture
    ):
        return next_state
    return current


# Reward function
def reward(state):
    if state == food_state:
        return 10
    if state in monster_states:
        return -8
    return -0.05


# Mean-squared error
def mean_squared_error(v1, v2):
    return np.mean((v1 - v2) ** 2)


# Calculating the value function from q-values using epsilon-greedy policy
def compute_value_function(q_table, epsilon):
    value_function = np.zeros(grid_size)
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            q_values = q_table[r, c]
            a_star = np.flatnonzero(q_values == np.max(q_values))  # Set of optimal actions
            policy = np.full(len(actions), epsilon / len(actions))
            policy[a_star] += (1 - epsilon) / len(a_star)
            value_function[r, c] = np.sum(policy * q_values)
    return value_function


# SARSA algorithm with steps and MSE computation
def sarsa_combined(episodes=1000):
    q_table = np.full((grid_size[0], grid_size[1], len(actions)), 10.0)
    q_table[food_state] = 0.0
    for state in monster_states + forbidden_furniture:
        q_table[state] = 0.0

    mse_per_episode = []
    steps_per_episode = []
    total_steps = 0

    for episode in range(episodes):
        state = (np.random.randint(0, 5), np.random.randint(0, 5))
        while state in forbidden_furniture or state == food_state:
            state = (np.random.randint(0, 5), np.random.randint(0, 5))
        action = actions[np.argmax(q_table[state])]

        steps = 0
        while state != food_state and steps < 1000:
            next_s = next_state_modified(state, action)
            reward_value = reward(next_s)
            next_a = actions[np.argmax(q_table[next_s])]

            q_table[state][action_index[action]] += alpha * (
                    reward_value
                    + gamma * q_table[next_s][action_index[next_a]]
                    - q_table[state][action_index[action]]
            )

            state = next_s
            action = next_a
            steps += 1
            total_steps += 1

        value_function = compute_value_function(q_table, epsilon)
        mse = mean_squared_error(value_function, true_v)
        mse_per_episode.append(mse)
        steps_per_episode.append(total_steps)

    return mse_per_episode, steps_per_episode, q_table


# Run SARSA for multiple runs
all_mse_per_episode = []
all_steps_per_episode = []

for run in range(runs):
    mse, steps, q_table_final = sarsa_combined(episodes)
    all_mse_per_episode.append(mse)
    all_steps_per_episode.append(steps)

# Computing average metrics
max_length_mse = max(len(mse) for mse in all_mse_per_episode)
avg_mse = np.zeros(max_length_mse)
for mse in all_mse_per_episode:
    avg_mse[:len(mse)] += mse
avg_mse /= runs

max_length_steps = max(len(steps) for steps in all_steps_per_episode)
avg_steps = np.zeros(max_length_steps)
for steps in all_steps_per_episode:
    avg_steps[:len(steps)] += steps
avg_steps /= runs

# Plot steps learning curve
plt.figure(figsize=(10, 6))
plt.plot(avg_steps, np.arange(len(avg_steps)))
plt.xlabel("Total Number of Actions Taken")
plt.ylabel("Number of Episodes Completed")
plt.title("Learning Curve: Timesteps vs. Episodes Completed")
plt.grid()
plt.show()

# Plot MSE learning curve
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(avg_mse)), avg_mse)
plt.xlabel("Number of Episodes")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Learning Curve: MSE vs. Episodes")
plt.grid()
plt.show()

# Compute and display greedy policy
greedy_policy = np.array(
    [
        [
            "G" if (r, c) == food_state else
            "X" if (r, c) in forbidden_furniture else
            actions[np.argmax(q_table_final[r, c])]
            for c in range(grid_size[1])
        ]
        for r in range(grid_size[0])
    ]
)

greedy_policy = pd.DataFrame(greedy_policy, index=None, columns=None)

# Print the DataFrame without indices or headers
with pd.option_context('display.show_dimensions', False, 'display.max_rows', None, 'display.max_columns', None):
    print("\nGreedy Policy:")
    print(greedy_policy.to_string(index=False, header=False))
