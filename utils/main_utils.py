from collections import OrderedDict


# Takes a string and returns and instance of an agent
# [env] is an instance of an environment
# [p] is a dictionary of agent parameters
def create_agent(agent_string, config):
    if agent_string == "DDPG": 
        from agents.DDPG import DDPG
        return DDPG(config)

    elif agent_string == "NAF":
        from agents.NAF import NAF
        return NAF(config)

    elif agent_string == "WireFitting":
        from agents.WireFitting import WireFitting
        return WireFitting(config)

    elif agent_string == "PICNN":
        from agents.PICNN import PICNN
        return PICNN(config)

    elif agent_string == "ActorExpert":
        from agents.ActorExpert import ActorExpert
        return ActorExpert(config)

    elif agent_string == "ActorExpert_Plus":
        from agents.ActorExpert_Plus import ActorExpert_Plus
        return ActorExpert_Plus(config)

    elif agent_string == "ActorExpert_PICNN":
        from agents.ActorExpert_PICNN import ActorExpert_PICNN
        return ActorExpert_PICNN(config)

    elif agent_string == "ActorExpert_Separate":
        from agents.ActorExpert_Separate import ActorExpert_Separate
        return ActorExpert_Separate(config)

    elif agent_string == "ActorExpert_Plus_Separate":
        from agents.ActorExpert_Plus_Separate import ActorExpert_Plus_Separate
        return ActorExpert_Plus_Separate(config)

    elif agent_string == "QT_OPT":
        from agents.QT_OPT import QT_OPT
        return QT_OPT(config)

    elif agent_string == "ActorCritic":
        from agents.ActorCritic import ActorCritic
        return ActorCritic(config)

    elif agent_string == "ActorCritic_Separate":
        from agents.ActorCritic_Separate import ActorCritic_Separate
        return ActorCritic_Separate(config)

    elif agent_string == "SoftActorCritic":
        from agents.SoftActorCritic import SoftActorCritic
        return SoftActorCritic(config)

    elif agent_string == "SoftQlearning":
        from agents.SoftQlearning import SoftQlearning
        return SoftQlearning(config)

    elif agent_string == "OptimalQ":
        from agents.OptimalQ import OptimalQ
        return OptimalQ(config)

    elif agent_string == 'ReverseKL':
        from agents.ReverseKL import ReverseKL
        return ReverseKL(config)

    elif agent_string == 'ForwardKL':
        from agents.ForwardKL import ForwardKL
        return ForwardKL(config)

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






