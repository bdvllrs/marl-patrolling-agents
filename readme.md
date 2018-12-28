# MARL for patrolling agents

Based on [Yasuyuki, S., Hirofumi, O., Tadashi, M., & Maya, H. (2015, May). Cooperative capture by multi-agent using reinforcement learning application for security patrol systems. In Control Conference (ASCC), 2015 10th Asian (pp. 1-6). IEEE](https://ieeexplore.ieee.org/document/7244682).

## Life-cycle of the Environment - Agent interactions
### Environment
The main environment is the `sim.Env` class.
To add a new agent use 
```python
env = sim.Env()
env.add_agent(agent: sim.Agent)
```

### Cycle
- `states = env.reset()` returns the initial state for every agent in the environment
- `states, actions, rewards, terminal = env.step()` returns the new states, actions, rewards for every agents.

The `step` method executes the agent's methods in this order:

For every agent:
- `action = sim.Agent.draw_action(observation)` where the observation is the position of all agents in the field of view
- `sim.Agent.set_position(position_from_action)`
- `sim.Agent.add_action_to_history(action)`

When this first loop is finished, rewards are computed for avery agent and transmitted:
- `sim.Agent.set_reward(reward)`
