from collections import OrderedDict

# Takes a string and returns and instance of an agent
# [env] is an instance of an environment
# [p] is a dictionary of agent parameters
def create_agent(agent_string, env, config, seed):
    if agent_string == "DDPG": 
        from agents.DDPG import DDPG
        return DDPG(env, config, seed)
    elif agent_string == "NAF":
        from agents.NAF import NAF
        return NAF(env, config, seed)
    elif agent_string == "Wire_fitting":
        from agents.WireFitting import Wire_fitting
        return Wire_fitting(env, config, seed)
    elif agent_string == "ICNN":
        from agents.ICNN import ICNN
        return ICNN(env, config, seed)

    elif agent_string == "AE_CCEM":
        from agents.AE_CCEM import AE_CCEM
        return AE_CCEM(env, config, seed)
    elif agent_string == "CriticAssistant_hydra":
        from agents.CriticAssistant_hydra import CriticAssistant_hydra
        return CriticAssistant_hydra(env, config, seed)


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






