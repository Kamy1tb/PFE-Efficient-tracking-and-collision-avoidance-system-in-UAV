__version__ = 'DQN'
__author__ = 'Pro Bagga_Razieh_First DRL'


from agent.agents.dqn.agentDQN import Agent as AgentDQN

from agent.helpers.rl_agent_template import *

def Agent(agent_type):

    if agent_type == "DQN":
        return AgentDQN()

    return None