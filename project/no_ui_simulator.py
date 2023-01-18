import numpy as np

import project.multiagent_configs as configs
import project.environment as single_agent_environment
from project.rendering_environment import Agent, MultiAgentRendering



if __name__ == '__main__':
    from project.solvers.q_learning_policy import q_policies

    initial_state = configs.LOCATIONS[configs.HOME].position
    initial_goal_label = configs.AREA_3
    obstacles = []
    all_craters, all_zones = configs.read_scenarios(1)

    suspected_craters = all_craters
    suspected_zones = all_zones
    craters_seen = set()
    zones_seen = set()
    suspected_crater_ids = set([c.id for c in suspected_craters])
    suspected_zone_ids = set([c.id for c in suspected_zones])

    available_policies = [configs.LOCATIONS[configs.HOME].policy,
                          configs.LOCATIONS[configs.AREA_1].policy,
                          configs.LOCATIONS[configs.AREA_2].policy,
                          configs.LOCATIONS[configs.AREA_3].policy]
    available_target_names = [configs.LOCATIONS[configs.HOME].name,
                              configs.LOCATIONS[configs.AREA_1].name,
                              configs.LOCATIONS[configs.AREA_2].name,
                              configs.LOCATIONS[configs.AREA_3].name]
    policy_container = q_policies(available_policies, available_target_names)

    a = Agent(1, (0, 0, 255), initial_state, initial_goal_label, available_policies, available_target_names, fov=10,
              _all_zones=all_zones, _suspected_zones=suspected_zone_ids,
              _all_craters=all_craters, _suspected_craters=suspected_crater_ids)

    rendering = MultiAgentRendering([1, 2, 3])
    goals = [configs.AREA_1, configs.AREA_2, configs.AREA_3]
    for i in range(100):
        # The events:
        #if i == 5:
        #    all_craters, all_zones = configs.read_scenarios(5)
        #    a.event(_all_craters=all_craters, _all_zones=all_zones)
        #    #rendering.change_event(new_craters=all_craters, new_zones=all_zones)

        # The dynamic assessment:
        assessment = a.dynamic_assess(goals)
        if assessment is not None:
            best_goal, labels, scores = assessment
            print("time: ", i, best_goal, labels[best_goal], scores[best_goal])
            rendering.change_event(new_goal_label=goals[best_goal])
            a.event(new_goal_label=goals[best_goal])
        should_stop = True
        states = []

        if a.is_done():
            break
        else:
            a.step()
            '''       
            for c in all_craters:
                if a.real_env.obstacle_in_fov(c):
                    craters_seen.add(c.id)
                    aware_craters = list(filter(lambda c: c.id in craters_suspected.union(craters_seen), all_craters))
                    rendering.change_event(new_craters=aware_craters)
                    rollout_env.change_event(new_craters=aware_craters)
            for z in all_zones:
                if a.real_env.obstacle_in_fov(z):
                    zones_seen.add(z.id)
                    aware_zones = list(filter(lambda z: z.id in zones_suspected.union(zones_seen), all_zones))
                    rendering.change_event(new_zones=aware_zones)
                    rollout_env.change_event(new_zones=aware_zones)
            '''
            should_stop = False
        rendering.state_update(a.get_state())
        rendering.change_event(new_zones=a.get_aware_zones(), new_craters=a.get_aware_craters())
        rendering.render(mode='human')
        #a.render()

        if should_stop:
            break