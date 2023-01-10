from multiprocessing import Process
import numpy as np

from project.backend import run_main
import project.multiagent_configs as configs

if __name__ == '__main__':
    condition = configs.Conditions.CONDITION_NO_GOA
    agent_ids = [1, 2]
    mission_id = 0
    subject_id = 123
    transition_probability = [0.8, 0.99]
    dust_transition_probability = [0.5, 0.8]
    crater_avoid_probability = [0, 0.95]
    dust_avoid_probability = [0, 0.95]
    triggering_threshold = 0.05
    processes = []
    randomized_transition_probability = transition_probability#np.random.choice(a=transition_probability, replace=False, size=len(transition_probability))
    randomized_dust_transition_probability = dust_transition_probability#np.random.choice(a=dust_transition_probability, replace=False, size=len(dust_transition_probability))
    randomized_crater_avoidance = crater_avoid_probability#np.random.choice(a=crater_avoid_probability, replace=False, size=len(crater_avoid_probability))
    randomized_dust_avoidance = dust_avoid_probability#np.random.choice(a=dust_avoid_probability, replace=False, size=len(dust_avoid_probability))
    for i in range(len(agent_ids)):
        a_id = agent_ids[i]
        p = Process(target=run_main, args=(a_id,
                                           mission_id,
                                           subject_id,
                                           configs.AGENT_COLOR[a_id],
                                           randomized_transition_probability[i],
                                           randomized_dust_transition_probability[i],
                                           randomized_crater_avoidance[i],
                                           randomized_dust_avoidance[i],
                                           triggering_threshold,
                                           condition))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()
