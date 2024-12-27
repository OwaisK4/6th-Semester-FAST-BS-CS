class GoalBasedAgent:
    def __init__(self):
        self.state = "start"
        self.goal = "finish"
        self.actions = ["move_forward", "turn_left", "turn_right", "grab", "release"]
        self.plan = []

    def perceive(self, environment):
        self.state = environment

    def think(self):
        if self.state == self.goal:
            self.plan = []
        elif not self.plan:
            self.plan = self.search_plan()

    def act(self):
        if self.plan:
            action = self.plan.pop(0)
            return action
        return "do_nothing"

    def search_plan(self):
        # A simple plan search algorithm
        return ["move_forward", "move_forward", "turn_left", "move_forward", "grab", "move_forward", "turn_right", "move_forward", "release", "move_forward"]

# Example usage
agent = GoalBasedAgent()

while True:
    environment = input("Enter the current state of the environment (start, finish, or obstacle): ")
    agent.perceive(environment)
    agent.think()
    action = agent.act()
    print("Agent performs action:", action)
    if agent.state == agent.goal:
        print("Goal reached!")
        break
