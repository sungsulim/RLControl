from collections import OrderedDict


# Takes a string and returns and instance of an agent
# [env] is an instance of an environment
# [p] is a dictionary of agent parameters
def create_agent(agent_string, env, config, seed):
    if agent_string == "DDPG": 
        from agents.DDPG import DDPG
        return DDPG(env, config, seed)
    elif agent_string == "DDPG_supervised":
        from agents.DDPG_supervised import DDPG_supervised
        return DDPG_supervised(env, config, seed)
    elif agent_string == "CriticAssistant_PG":
        from agents.CriticAssistant_PG import CriticAssistant_PG
        return CriticAssistant_PG(env, config, seed)
    elif agent_string == "CriticAssistant":
        from agents.CriticAssistant import CriticAssistant
        return CriticAssistant(env, config, seed)
    elif agent_string == "CriticAssistant_CEM":
        from agents.CriticAssistant_CEM import CriticAssistant_CEM
        return CriticAssistant_CEM(env, config, seed)
    elif agent_string == "CriticAssistant_hydra":
        from agents.CriticAssistant_hydra import CriticAssistant_hydra
        return CriticAssistant_hydra(env, config, seed)
    elif agent_string == "CEM_hydra":
        from agents.CEM_hydra import CEM_hydra
        return CEM_hydra(env, config, seed)
    elif agent_string == "CEM_hydra_multimodal":
        from agents.CEM_hydra_multimodal import CEM_hydra_multimodal
        return CEM_hydra_multimodal(env, config, seed)

    elif agent_string == "Wire_fitting":
        from agents.WireFitting import Wire_fitting
        return Wire_fitting(env, config, seed)
    elif agent_string == "NAF":
        from agents.NAF import NAF
        return NAF(env, config, seed)

    elif agent_string == "Omniscient":
        from agents.Omniscient import Omniscient
        return Omniscient(env, config, seed)
    elif agent_string == "Ignorant":
        from agents.Ignorant import Ignorant
        return Ignorant(env, config, seed)
    elif agent_string == "Ignorant_CA":
        from agents.Ignorant_CA import Ignorant_CA
        return Ignorant_CA(env, config, seed)
    elif agent_string == "GreedyGQ":
        from agents.GreedyGQ import GreedyGQ
        return GreedyGQ(env, config, seed)
    elif agent_string == "ICNN":
        from agents.ICNN import ICNN
        return ICNN(env, config, seed)

    else:
        print("Don't know this agent")
        exit(0)

# takes a dictionary where each key maps to a list of different parameter choices for that key
# also takes an index where 0 < index < combinations(parameters)
# The algorithm does wrap back around if index > combinations(parameters), so setting index higher allows for multiple runs of same parameter settings
# Index is not necessarily in the order defined in the json file.
def get_sweep_parameters(parameters, index):
    out = OrderedDict()
    accum = 1
    for key in parameters:
        num = len(parameters[key])
        out[key] = parameters[key][int(index / accum) % num]
        accum *= num
    return (out, accum)


