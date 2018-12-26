import sim

patrols = [sim.Patrol("Patrol " + str(k)) for k in range(3)]
target = sim.Target()

env = sim.World(width=100, height=100)
env.max_iterations = 10

for patrol in patrols:
    env.add_actor(patrol)
env.add_actor(target)

states = env.reset()
env.draw_board()
terminal = False
while not terminal:
    states, actions, terminal = env.step()
    env.draw_board()

