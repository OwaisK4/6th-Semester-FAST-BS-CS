import random

def display(room):
    print(room)

room = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
]

print("All the rooms are dirty")
display(room)

x = 0
y = 0

while x < 4:
    while y < 4:
        room[x][y] = random.choice([0, 1])
        y += 1
    x += 1
    y = 0

print("Before cleaning, the room has these random dirt locations:")
display(room)

x = 0
y = 0
z = 0

while x < 4:
    while y < 4:
        if room[x][y] == 1:
            print("Vacuuming at location:", x, y)
            room[x][y] = 0
            print("Cleaned at location:", x, y)
            z += 1
        y += 1
    x += 1
    y = 0

pro = (100 - ((z / 16) * 100))
print("Room is clean now, Thanks for using : 3710933")
display(room)
print('performance=', pro, '%')

# another way

class ModelBasedReflexAgent:
    def __init__(self):
        self.location = "A"
        self.status = "Dirty"
        self.model = {
            "A": "Dirty",
            "B": "Dirty"
        }
        self.actions = []

    def perceive(self, location, status):
        self.location = location
        self.status = status
        self.model[location] = status

    def think(self):
        if self.status == "Dirty":
            self.actions.append("Suck")
            self.model[self.location] = "Clean"
        else:
            if self.model["A"] == "Dirty" or self.model["B"] == "Dirty":
                if self.location == "A":
                    self.actions.append("Right")
                else:
                    self.actions.append("Left")
            else:
                if self.location == "A":
                    self.actions.append("Right")
                    self.model["A"] = "Dirty"
                else:
                    self.actions.append("Left")
                    self.model["B"] = "Dirty"

    def act(self):
        if self.actions:
            action = self.actions.pop(0)
            return action
        return "NoOp"

# Example usage
agent = ModelBasedReflexAgent()

while True:
    location = input("Enter the current location of the agent (A or B): ")
    status = input("Enter the status of the current location (Dirty or Clean): ")
    agent.perceive(location, status)
    agent.think()
    action = agent.act()
    print("Agent performs action:", action)
    if action == "NoOp":
        break


################## Q-learning (REINFORCEMENT LEARNING DONOT UUSE THIS IN EXAM)
    import numpy as np

class ModelBasedLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.model = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def select_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(len(self.q_table[state]))
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state, action] = new_q

        # Update model
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (reward, next_state)

    def simulate(self, state, action):
        if state in self.model and action in self.model[state]:
            reward, next_state = self.model[state][action]
            return reward, next_state
        return 0, state  # Default to no change in state and no reward

# Simple Grid World Environment
class GridWorld:
    def __init__(self):
        self.n_states = 9
        self.n_actions = 4
        self.state = 0  # Start at state 0
        self.done = False

    def reset(self):
        self.state = 0
        self.done = False
        return self.state

    def step(self, action):
        if self.state == 8:  # Terminal state
            self.done = True
            return 8, 0, self.done

        if action == 0:  # Up
            self.state -= 3
        elif action == 1:  # Down
            self.state += 3
        elif action == 2 and self.state % 3 != 0:  # Left
            self.state -= 1
        elif action == 3 and self.state % 3 != 2:  # Right
            self.state += 1

        reward = -1
        if self.state == 8:
            reward = 10  # Goal state

        return self.state, reward, self.done

# Training the agent
env = GridWorld()
agent = ModelBasedLearningAgent(env.n_states, env.n_actions)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state

# Testing the agent
state = env.reset()
done = False
while not done:
    action = agent.select_action(state)
    reward, next_state = agent.simulate(state, action)
    state = next_state
    print("Agent at state:", state)
