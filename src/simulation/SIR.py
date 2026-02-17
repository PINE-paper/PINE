import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx


def run_SIR(G, S, beta=0.05, gamma=0.2, stable_steps=10):
    '''
    Input: G -- networkx directed graph
    S -- initial seed set of nodes
    '''

    model = ep.SIRModel(G)
    config = mc.Configuration()

    config.add_model_parameter('beta', beta) 
    config.add_model_parameter('gamma', gamma)  

    config.add_model_initial_configuration("Infected", S)

    model.set_initial_status(config)

    iteration_num = 0
    infected_counts = []

    while True:
    # while iteration_num <= 100:
        iteration = model.iteration() 
        iteration_num += 1

        status = iteration['status']

        infected_count = sum(1 for state in status.values() if state == 1)
        infected_counts.append(infected_count)

        if iteration_num >= stable_steps:
            recent_counts = infected_counts[-stable_steps:]
            if len(set(recent_counts)) == 1:
                break
    
        if (infected_count == 0):
            break

    influence_count = sum(1 for state in model.status.values() if (state == 1) or (state == 2))
    return influence_count 
