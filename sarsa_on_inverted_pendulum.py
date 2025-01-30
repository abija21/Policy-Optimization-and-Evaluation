import numpy as np
import matplotlib.pyplot as plt

# Constants
ACTION_BOUNDS = (-2, 2)  # Continuous action bounds (torque)
MAX_SPEED = 8  # Maximum angular velocity
MAX_TORQUE = 2  # Maximum torque
G = 10  # Gravity
L = 1  # Length of pendulum
M = 1  # Mass of pendulum
DT = 0.05  # Time step
MAX_STEPS = 200  # Maximum steps per episode

ALPHA = 0.1  # Learning rate
GAMMA = 0.99  # Discount factor
EPSILON = 0.1  # Epsilon for epsilon-greedy policy

NUM_ACTIONS = 5  # Number of discrete actions to represent the continuous space
NUM_BINS = 5  # Number of bins for state discretization
NUM_RUNS = 20  # Number of independent runs
NUM_EPISODES = 1000  # Total number of episodes

# Discretize the action space
def discretize_action_space(bounds, num_actions):
    return np.linspace(bounds[0], bounds[1], num_actions)


DISCRETE_ACTIONS = discretize_action_space(ACTION_BOUNDS, NUM_ACTIONS)


# Transition dynamics
def step_deterministic(action, omega, omega_dot):
    # Convert action index to torque value
    torque = DISCRETE_ACTIONS[action]

    # Compute angular acceleration
    angular_acceleration = (3 * G / (2 * L)) * np.sin(omega) + (3 * torque) / (M * L ** 2)

    # Update angular velocity and clip to MAX_SPEED
    omega_dot_new = np.clip(omega_dot + angular_acceleration * DT, -MAX_SPEED, MAX_SPEED)

    # Update angle
    omega_new = omega + omega_dot_new * DT

    # Compute reward 
    reward = -np.abs(((omega_new + np.pi) % (2 * np.pi)) - np.pi)

    return omega_new, omega_dot_new, reward


# Discretize state
def discretize_state(omega, omega_dot, bins):
    omega = ((omega + np.pi) % (2 * np.pi)) - np.pi
    omega_dot = np.clip(omega_dot, -MAX_SPEED, MAX_SPEED)
    omega_bins = np.linspace(-np.pi, np.pi, bins)
    omega_dot_bins = np.linspace(-MAX_SPEED, MAX_SPEED, bins)
    b_omega = min(max(np.digitize([omega], omega_bins)[0] - 1, 0), bins - 1)
    b_omega_dot = min(max(np.digitize([omega_dot], omega_dot_bins)[0] - 1, 0), bins - 1)
    return b_omega, b_omega_dot


# Epsilon-greedy policy
def epsilon_greedy_policy(Q, state, epsilon=EPSILON):
    if np.random.rand() < epsilon:
        return np.random.choice(NUM_ACTIONS) 
    return np.argmax(Q[state])


# SARSA Algorithm 
def sarsa_algorithm_with_multiple_runs(bins):
    all_returns = np.zeros((NUM_RUNS, NUM_EPISODES))  

    for run in range(NUM_RUNS):
        Q = np.zeros((bins, bins, NUM_ACTIONS))  
        returns = []

        for episode in range(NUM_EPISODES):
            omega, omega_dot = 0, 0  
            state = discretize_state(omega, omega_dot, bins)
            action = epsilon_greedy_policy(Q, state)

            total_return = 0
            for _ in range(MAX_STEPS):
                omega_new, omega_dot_new, reward = step_deterministic(action, omega, omega_dot)
                state_new = discretize_state(omega_new, omega_dot_new, bins)
                action_new = epsilon_greedy_policy(Q, state_new)

                Q[state][action] += ALPHA * (
                        reward + GAMMA * Q[state_new][action_new] - Q[state][action]
                )

                state, action = state_new, action_new
                omega, omega_dot = omega_new, omega_dot_new
                total_return += reward

            returns.append(total_return)

        all_returns[run] = returns

    return all_returns


# Run the SARSA algorithm with multiple runs
all_returns_with_dynamics = sarsa_algorithm_with_multiple_runs(NUM_BINS)

# Calculate mean and standard deviation across runs
mean_returns = np.mean(all_returns_with_dynamics, axis=0)
std_returns = np.std(all_returns_with_dynamics, axis=0)

# Plot learning curve for the updated dynamics with multiple runs
plt.figure(figsize=(10, 6))
plt.plot(range(NUM_EPISODES), mean_returns, label="Average Return")
plt.fill_between(
    range(NUM_EPISODES),
    mean_returns - std_returns,
    mean_returns + std_returns,
    alpha=ALPHA,
    label="Standard Deviation",
)
plt.xlabel('Number of Episodes')
plt.ylabel('Average Return')
plt.title('Learning Curve for SARSA on Inverted Pendulum')
plt.legend()
plt.grid()
plt.show()
