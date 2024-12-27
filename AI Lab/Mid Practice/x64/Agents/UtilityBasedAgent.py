class UtilityBasedAgent:
    def __init__(self):
        self.location = "A"
        self.status = "Dirty"
        self.utility = {
            "A": {"Dirty": -1, "Clean": 10},
            "B": {"Dirty": -1, "Clean": 10}
        }

    def perceive(self, location, status):
        self.location = location
        self.status = status

    def think(self):
        if self.status == "Dirty":
            return "Suck"
        else:
            if self.utility["A"]["Dirty"] > self.utility["B"]["Dirty"]:
                return "Right"
            else:
                return "Left"

    def act(self):
        action = self.think()
        return action

# Example usage
agent = UtilityBasedAgent()

while True:
    location = input("Enter the current location of the agent (A or B): ")
    status = input("Enter the status of the current location (Dirty or Clean): ")
    agent.perceive(location, status)
    action = agent.act()
    print("Agent performs action:", action)
    if action == "NoOp":
        break
