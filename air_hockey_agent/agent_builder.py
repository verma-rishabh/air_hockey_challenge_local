from air_hockey_agent.PPO import PPO_agent 


def build_agent(env_info,state_dim=12, action_dim=6, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=256, eps_clip=0.2, has_continuous_action_space=True, action_std_init=0.6):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    agent_id = 1
    agent = PPO_agent(env_info,agent_id,state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6)
    # agent.load("models/TD3-v2_air-hockey_f")
    agent.load("PPO_preTrained/air-hockey-hit/PPO_air-hockey-hit_0_2.pth")

    return agent
