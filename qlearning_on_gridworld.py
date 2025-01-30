import numpy as np
import matplotlib.pyplot as plt

# Define environment parameters
grid_size = 5
forbidden_furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]
monsters = [(0, 3), (4, 1)]
food_state = (4, 4)
gamma = 0.925
reward_food = 10
reward_monster = -8
reward_step = -0.05

actions = ['AU', 'AD', 'AL', 'AR']
action_effects = {
    'AU': (-1, 0),
    'AD': (1, 0),
    'AL': (0, -1),
    'AR': (0, 1),
}

true_optimal_v = np.array([
    [2.6638, 2.9969, 2.8117, 3.6671, 4.8497],
    [2.9713, 3.5101, 4.0819, 4.8497, 7.1648],
    [2.5936, 0.0000, 0.0000, 0.0000, 8.4687],
    [2.0993, 1.0850, 0.0000, 8.6097, 9.5269],
    [1.0850, 4.9466, 8.4687, 9.5269, 0.0000]
])

alpha_initial = 0.5
epsilon = 0.2
num_episodes = 1000
max_steps_per_episode = 50


def initialize_q_function():
    q = np.full((grid_size, grid_size, len(actions)), 10.0)
    for r, c in forbidden_furniture + monsters + [food_state]:
        q[r, c, :] = 0.0
    return q


def epsilon_greedy_policy(q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(actions))
    else:
        return np.argmax(q[state])


def environment_step(state, action):
    r, c = state
    dr, dc = action_effects[actions[action]]
    prob = np.random.rand()

    if prob <= 0.7:
        next_state = (r + dr, c + dc)
    elif prob <= 0.82:
        next_state = (r - dc, c + dr)
    elif prob <= 0.94:
        next_state = (r + dc, c - dr)
    else:
        next_state = (r, c)

    nr, nc = next_state
    if not (0 <= nr < grid_size and 0 <= nc < grid_size) or (nr, nc) in forbidden_furniture:
        next_state = state

    if next_state == food_state:
        reward = reward_food
        terminal = True
    elif next_state in monsters:
        reward = reward_monster
        terminal = True
    else:
        reward = reward_step
        terminal = False

    return next_state, reward, terminal


def q_learning():
    q = initialize_q_function()
    alpha = alpha_initial
    episodes_completed = []
    steps_per_episode = []
    total_steps = 0

    for episode in range(num_episodes):
        state = (np.random.randint(grid_size), np.random.randint(grid_size))
        while state in forbidden_furniture + monsters + [food_state]:
            state = (np.random.randint(grid_size), np.random.randint(grid_size))

        for step in range(max_steps_per_episode):
            total_steps += 1
            action = epsilon_greedy_policy(q, state, epsilon)
            next_state, reward, terminal = environment_step(state, action)
            best_next_action = np.argmax(q[next_state])
            q[state][action] += alpha * (reward + gamma * q[next_state][best_next_action] - q[state][action])
            state = next_state
            if terminal:
                break

        episodes_completed.append(episode + 1)
        steps_per_episode.append(total_steps)
        alpha = alpha_initial / (1 + 0.01 * episode)

    return q, steps_per_episode, episodes_completed


def compute_greedy_policy(q):
    policy = []
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            if (r, c) in forbidden_furniture:
                row.append('X')
            elif (r, c) == food_state:
                row.append('G')
            else:
                row.append(actions[np.argmax(q[r, c])])
        policy.append(row)
    return np.array(policy)


def compute_value_function(q, policy):
    v = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            v[r, c] = np.sum(policy[r, c] * q[r, c])
    return v


def compute_mse(v1, v2):
    return np.mean((v1 - v2) ** 2)


def q_learning_with_mse():
    q = initialize_q_function()
    alpha = alpha_initial
    mse_per_episode = []

    for episode in range(num_episodes):
        policy = np.zeros_like(q, dtype=float)
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) in forbidden_furniture + [food_state]:
                    continue
                max_actions = np.flatnonzero(q[r, c] == np.max(q[r, c]))
                for a in range(len(actions)):
                    if a in max_actions:
                        policy[r, c, a] = (1 - epsilon) / len(max_actions) + epsilon / len(actions)
                    else:
                        policy[r, c, a] = epsilon / len(actions)

        v = compute_value_function(q, policy)
        mse = compute_mse(v, true_optimal_v)
        mse_per_episode.append(mse)

        state = (np.random.randint(grid_size), np.random.randint(grid_size))
        while state in forbidden_furniture + monsters + [food_state]:
            state = (np.random.randint(grid_size), np.random.randint(grid_size))

        for step in range(max_steps_per_episode):
            action = epsilon_greedy_policy(q, state, epsilon)
            next_state, reward, terminal = environment_step(state, action)
            best_next_action = np.argmax(q[next_state])
            q[state][action] += alpha * (reward + gamma * q[next_state][best_next_action] - q[state][action])
            state = next_state
            if terminal:
                break

        alpha = alpha_initial / (1 + 0.01 * episode)

    return q, mse_per_episode


# Learning Curve
runs = 20
all_steps = []
for _ in range(runs):
    _, steps, _ = q_learning()
    all_steps.append(steps)

average_steps = np.mean(all_steps, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_episodes + 1), average_steps)
plt.xlabel('Episodes')
plt.ylabel('Total Steps')
plt.title('Learning Curve: Q-Learning in Cat-vs-Monsters Domain')
plt.grid()
plt.show()

# MSE Graph
all_mse = []
for _ in range(runs):
    _, mse_curve = q_learning_with_mse()
    all_mse.append(mse_curve)

average_mse = np.mean(all_mse, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_episodes + 1), average_mse)
plt.xlabel('Episodes')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Episodes in Cat-vs-Monsters Domain')
plt.grid()
plt.show()

# Greedy Policy
final_q, _, _ = q_learning()
greedy_policy = compute_greedy_policy(final_q)
print("Greedy Policy:")
print(greedy_policy)
