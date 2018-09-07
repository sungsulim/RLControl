from collections import OrderedDict


# Takes a string and returns and instance of an agent
# [env] is an instance of an environment
# [p] is a dictionary of agent parameters
def create_agent(agent_string, config, seed):
    if agent_string == "DDPG": 
        from agents.DDPG import DDPG
        return DDPG(config, seed)

    elif agent_string == "NAF":
        from agents.NAF import NAF
        return NAF(config, seed)

    elif agent_string == "WireFitting":
        from agents.WireFitting import WireFitting
        return WireFitting(config, seed)

    elif agent_string == "ICNN":
        from agents.ICNN import ICNN
        return ICNN(config, seed)

    elif agent_string == "ActorExpert":
        from agents.ActorExpert import ActorExpert
        return ActorExpert(config, seed)

    elif agent_string == "ActorExpert_Plus":
        from agents.ActorExpert_Plus import ActorExpert_Plus
        return ActorExpert_Plus(config, seed)

    # elif agent_string == "AE_CCEM_separate":
    #     from agents.AE_CCEM_separate import AE_CCEM_separate
    #     return AE_CCEM_separate(env, config, seed)
    # elif agent_string == "AE_Supervised_separate":
    #     from agents.AE_Supervised_separate import AE_Supervised_separate
    #     return AE_Supervised_separate(env, config, seed)

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






