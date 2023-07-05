from air_hockey_agent.TD3 import TD3_agent 
# from air_hockey_agent.OurDDPG import DDPG_agent 


def build_agent(env_info,agent_id=1):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    agent = TD3_agent(env_info,agent_id)
    # agent.load("models/default/TD3_air-hockey-defend_0")
    return agent
