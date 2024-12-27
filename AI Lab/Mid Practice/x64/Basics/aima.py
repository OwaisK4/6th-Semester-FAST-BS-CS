class TrivialVacuumEnvironment(Environment):
# """This environment has two locations, A and B. Each can be Dirty or
# Clean. The agent perceives its location and the location's status. This
# serves as an example of how to implement a simple Environment."""

    def init (self): 
        super().init()
        self.status = {loc_A: random.choice(['Clean', 'Dirty']), loc_B:random.choice(['Clean', 'Dirty'])}

    def thing_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,TableDrivenVacuumAgent, ModelBasedVacuumAgent]
    
    def percept(self, agent):
# """Returns the agent's location, and the location status
# (Dirty/Clean)."""
        return (agent.location, self.status[agent.location])

    def execute_action(self, agent, action):
# """Change agent's location and/or location's status; track performance.
# Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right': 
            agent.location = loc_B 
            agent.performance -= 1
        elif action == 'Left': 
            agent.location = loc_A
            agent.performance -= 1 
        elif action == 'Suck':
            self.status[agent.location] == 'Dirty': agent.performance += 10 
            self.status[agent.location] = 'Clean'
    
    def default_location(self, thing):
# """Agents start in either location at random."""
        return random.choice([loc_A, loc_B]):