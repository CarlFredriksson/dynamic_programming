import numpy as np

def policy_evaluation(dynamics, policy, state_values):
    num_states = len(state_values)
    new_state_values = np.zeros(num_states)
    for state in range(num_states):
        value = 0
        num_actions = len(policy[state])
        for action in range(num_actions):
            new_state, reward = dynamics[(state, action)]
            value += policy[state, action] * (reward + state_values[new_state])
        new_state_values[state] = value
    return new_state_values

def policy_evaluation_sweep(dynamics, policy, state_values):
    num_states = len(state_values)
    for state in range(num_states):
        value = 0
        num_actions = len(policy[state])
        for action in range(num_actions):
            new_state, reward = dynamics[(state, action)]
            value += policy[state, action] * (reward + state_values[new_state])
        state_values[state] = value

def value_iteration_sweep(dynamics, policy, state_values):
    num_states = len(state_values)
    for state in range(num_states):
        max_value = -np.inf
        num_actions = len(policy[state])
        for action in range(num_actions):
            new_state, reward = dynamics[(state, action)]
            value = reward + state_values[new_state]
            if value > max_value:
                max_value = value
                policy[state] = np.eye(num_actions)[action]

if __name__ == "__main__":
    # Actions: 0=up, 1=down, 2=right, 3=left
    # Deterministic dynamics: (state, action) -> (new_state, reward)
    DYNAMICS = {
        (0, 0): (0, -1),
        (0, 1): (4, -1),
        (0, 2): (1, -1),
        (0, 3): (14, -1),
        (1, 0): (1, -1),
        (1, 1): (5, -1),
        (1, 2): (2, -1),
        (1, 3): (0, -1),
        (2, 0): (2, -1),
        (2, 1): (6, -1),
        (2, 2): (2, -1),
        (2, 3): (1, -1),
        (3, 0): (14, -1),
        (3, 1): (7, -1),
        (3, 2): (4, -1),
        (3, 3): (3, -1),
        (4, 0): (0, -1),
        (4, 1): (8, -1),
        (4, 2): (5, -1),
        (4, 3): (3, -1),
        (5, 0): (1, -1),
        (5, 1): (9, -1),
        (5, 2): (6, -1),
        (5, 3): (4, -1),
        (6, 0): (2, -1),
        (6, 1): (10, -1),
        (6, 2): (6, -1),
        (6, 3): (5, -1),
        (7, 0): (3, -1),
        (7, 1): (11, -1),
        (7, 2): (8, -1),
        (7, 3): (7, -1),
        (8, 0): (4, -1),
        (8, 1): (12, -1),
        (8, 2): (9, -1),
        (8, 3): (7, -1),
        (9, 0): (5, -1),
        (9, 1): (13, -1),
        (9, 2): (10, -1),
        (9, 3): (8, -1),
        (10, 0): (6, -1),
        (10, 1): (14, -1),
        (10, 2): (10, -1),
        (10, 3): (9, -1),
        (11, 0): (7, -1),
        (11, 1): (11, -1),
        (11, 2): (12, -1),
        (11, 3): (11, -1),
        (12, 0): (8, -1),
        (12, 1): (12, -1),
        (12, 2): (13, -1),
        (12, 3): (11, -1),
        (13, 0): (9, -1),
        (13, 1): (13, -1),
        (13, 2): (14, -1),
        (13, 3): (12, -1),
        (14, 0): (14, 0),
        (14, 1): (14, 0),
        (14, 2): (14, 0),
        (14, 3): (14, 0),
    }
    NUM_STATES = 15
    NUM_ACTIONS = 4

    # Non-sweeping policy evaluation
    print("******Non-sweeping policy evaluation******")
    policy = np.ones((NUM_STATES, NUM_ACTIONS)) / NUM_ACTIONS
    state_values = np.zeros(NUM_STATES)
    print(state_values)
    for _ in range(10):
        state_values = policy_evaluation(DYNAMICS, policy, state_values)
        print(state_values)
    print("--------------------------------")
    print(policy)
    value_iteration_sweep(DYNAMICS, policy, state_values)
    print(policy)

    # Sweeping policy evaluation
    print("******Sweeping policy evaluation******")
    policy = np.ones((NUM_STATES, NUM_ACTIONS)) / NUM_ACTIONS
    state_values = np.zeros(NUM_STATES)
    print(state_values)
    for _ in range(10):
        policy_evaluation_sweep(DYNAMICS, policy, state_values)
        print(state_values)
    print("--------------------------------")
    print(policy)
    value_iteration_sweep(DYNAMICS, policy, state_values)
    print(policy)
