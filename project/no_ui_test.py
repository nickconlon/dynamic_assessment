from scipy.stats import fisher_exact
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import project.multiagent_configs as configs
from project.rendering_environment import Agent, MultiAgentRendering
from project.environment import Environment

sns.set_theme()


def run_difficulty(num_runs, craters_run, craters_event1, craters_event2, zones_run, zones_event1, zones_event2):
    initial_state = configs.LOCATIONS[configs.HOME].position

    available_policies = [configs.LOCATIONS[configs.HOME].policy,
                          configs.LOCATIONS[configs.AREA_1].policy,
                          configs.LOCATIONS[configs.AREA_2].policy,
                          configs.LOCATIONS[configs.AREA_3].policy]
    available_target_names = [configs.LOCATIONS[configs.HOME].name,
                              configs.LOCATIONS[configs.AREA_1].name,
                              configs.LOCATIONS[configs.AREA_2].name,
                              configs.LOCATIONS[configs.AREA_3].name]

    a1 = Agent(1, (0, 0, 255), initial_state, configs.AREA_1, available_policies, available_target_names,
               _all_zones=craters_run, _all_craters=zones_run, fov=10)

    rendering = MultiAgentRendering([1])
    rendering.agent_FOV = 20

    zones = np.zeros(num_runs, dtype=int)
    craters = np.zeros(num_runs, dtype=int)
    deliveries = np.zeros(num_runs, dtype=int)
    assessments = np.zeros(num_runs, dtype=int)
    confidences = []
    all_confidences = []
    for delivery in range(num_runs):
        print('Round {}/{}'.format(delivery, num_runs))
        new_craters = craters_run[delivery]
        new_zones = zones_run[delivery]
        all_confidence = {}
        conf_test = np.ones((3, 2)) * -1
        rendering.reset()
        a1.reset(initial_state)
        suspected_crater_ids = set([c.id for c in new_craters])
        suspected_zone_ids = set([c.id for c in new_zones])
        a1.event(_all_zones=new_zones, _all_craters=new_craters,
                 _suspected_craters=suspected_crater_ids, _suspected_zones=suspected_zone_ids)

        #######
        # Choosing the goal here based on self-confidence
        goal = configs.AREA_3
        a1.event(new_goal_label=goal)
        out = a1.choose_goal([goal])
        all_confidence[0] = [delivery, out[2][0]]
        conf_test[0] = [0, out[2][0]]
        done = False
        for i in range(35):
            if i == 10:
                a1.real_env.change_event(new_craters=craters_event1[delivery], new_zones=zones_event1[delivery])
            if i == 30:
                a1.real_env.change_event(new_craters=craters_event2[delivery], new_zones=zones_event2[delivery])
            if a1.is_done():
                done = True
            else:
                a1.step()
                out = a1.dynamic_assess([goal])
                if out is not None:
                    # save off the post-event assessment
                    if i >= 10 and conf_test[1][0] < 0:
                        conf_test[1] = [i, out[2][0]]
                    elif i >= 30 and conf_test[2][0] < 0:
                        conf_test[2] = [i, out[2][0]]
                    all_confidence[i] = [delivery, out[2][0]]

            #rendering.state_update(a1.get_state())
            #rendering.change_event(new_zones=a1.get_aware_zones(), new_craters=a1.get_aware_craters())
            #rendering.render(mode='human')
            if done:
                if a1.craters < 5:
                    deliveries[delivery] = 1
                break
        zones[delivery] = a1.zones
        craters[delivery] = a1.craters
        assessments[delivery] = a1.assessments
        all_confidences.append(all_confidence)
        if -1 not in conf_test:
            confidences.append(conf_test)
        print('  zones:   ', a1.zones)
        print('  craters: ', a1.craters)
        print('  deliveries: ', sum(deliveries))
        print('  assessments: ', a1.assessments)
        print('  confidences: ', confidences[delivery])
    return zones, craters, deliveries, assessments, confidences, all_confidences


def run(run_type, num_runs, craters_run, craters_event, zones_run, zones_event):
    initial_state = configs.LOCATIONS[configs.HOME].position

    available_policies = [configs.LOCATIONS[configs.HOME].policy,
                          configs.LOCATIONS[configs.AREA_1].policy,
                          configs.LOCATIONS[configs.AREA_2].policy,
                          configs.LOCATIONS[configs.AREA_3].policy]
    available_target_names = [configs.LOCATIONS[configs.HOME].name,
                              configs.LOCATIONS[configs.AREA_1].name,
                              configs.LOCATIONS[configs.AREA_2].name,
                              configs.LOCATIONS[configs.AREA_3].name]

    a1 = Agent(1, (0, 0, 255), initial_state, configs.AREA_1, available_policies, available_target_names,
               _all_obstacles=[], _all_zones=craters_run, _all_craters=zones_run, fov=10)

    rendering = MultiAgentRendering([1])
    rendering.agent_FOV = 10

    zones = np.zeros(num_runs, dtype=int)
    craters = np.zeros(num_runs, dtype=int)
    deliveries = np.zeros(num_runs, dtype=int)
    assessments = np.zeros(num_runs, dtype=int)
    for delivery in range(num_runs):
        print('Round {}/{}'.format(delivery, num_runs))
        new_craters = craters_run[delivery]
        new_zones = zones_run[delivery]
        rendering.reset()
        a1.reset(initial_state)
        a1.real_env.agent_FOV = 10
        suspected_crater_ids = set([c.id for c in new_craters])
        suspected_zone_ids = set([c.id for c in new_zones])
        a1.event(_all_zones=new_zones, _all_craters=new_craters,
                 _suspected_craters=suspected_crater_ids, _suspected_zones=suspected_zone_ids)

        #######
        # Choosing the goal here based on self-confidence
        goals = [configs.AREA_1, configs.AREA_2, configs.AREA_3]
        if run_type == 'static' or run_type == 'dynamic-static' or 'triggered' in run_type:
            best_zone_idx, _, _ = a1.choose_goal(goals)
        elif run_type == 'random' or run_type == 'dynamic-random':
            best_zone_idx = np.random.randint(0, 3)
        elif run_type == 'human' or run_type == 'dynamic-human':
            rendering.render(mode='human')
            _, scores, _ = a1.choose_goal(goals)
            print(scores)
            best_zone_idx = int(input('Choose goal:'))
        print(" chosen goal is {}".format(goals[best_zone_idx]))
        a1.event(new_goal_label=goals[best_zone_idx])

        done = False
        for i in range(100):
            if a1.is_done():
                done = True
            else:
                a1.step()
            if 'triggered' in run_type and not done:
                trigger = a1.dynamic_assess(goals)
                if trigger is not None:
                    a1.event(new_goal_label=goals[trigger[0]])
            if 'dynamic' in run_type and not done:
                if i == 10:
                    a1.real_env.change_event(new_craters=craters_event[delivery], new_zones=zones_event[delivery])

            #rendering.state_update(a1.get_state())
            #rendering.change_event(new_zones=a1.get_aware_zones(), new_craters=a1.get_aware_craters())
            #rendering.render(mode='human')
            if done:
                if a1.craters < 5:
                    deliveries[delivery] = 1
                break
        zones[delivery] = a1.zones
        craters[delivery] = a1.craters
        assessments[delivery] = a1.assessments
        print('  zones:   ', a1.zones)
        print('  craters: ', a1.craters)
        print('  deliveries: ', sum(deliveries))
        print('  assessments: ', a1.assessments)
    return zones, craters, deliveries, assessments


def write_csv(condition, zones, craters, deliveries, assessments):
    header = 'condition,dust,craters,deliveries,assessments'
    with open(condition + '.csv', 'w') as file:
        file.write(header + '\n')
        for (z, c, d, a) in zip(zones, craters, deliveries, assessments):
            file.write('{},{},{},{},{}\n'.format(condition, z, c, d, a))


def task_difficulty():
    num_runs = 130
    craters1 = []
    zones1 = []
    craters2 = []
    zones2 = []
    craters_empty = []
    zones_empty = []
    for i in range(num_runs):
        srange = (4, 8)

        c1, z1 = configs.create_new_craters(100,
                                            lxrange=(15, 30),
                                            lyrange=(15, 35),
                                            sxrange=srange,
                                            syrange=srange)
        c2, z2 = configs.create_new_craters(100,
                                            lxrange=(25, 40),
                                            lyrange=(25, 45),
                                            sxrange=srange,
                                            syrange=srange)
        craters1.append(c1)
        zones1.append([])
        craters2.append(c2)
        zones2.append([])
        craters_empty.append([])
        zones_empty.append([])

    # easy -> hard -> easy
    zones, craters, deliveries, assessments, confidence, all_confidences = run_difficulty(num_runs,
                                                                         craters_empty, craters1, craters_empty,
                                                                         zones_empty, zones1, zones_empty)
    conditions = []
    conf = []
    time = []
    runid = []
    idx = 0
    confidence = confidence[:100]
    for _run in confidence:
        idx+=1
        for event in _run:
            conditions.append('easy->hard')
            time.append(event[0])
            conf.append(event[1])
            runid.append(idx)
    header = "condition, confidence, time, runid"
    with open('difficulty_all.csv', 'w') as file:
        file.write(header + '\n')
        for (cond, c, t, rid) in zip(conditions, conf, time, runid):
            file.write('{},{},{},{}\n'.format(cond, c, t, rid))

    with open('difficulty_all_conf.csv', 'w') as file:
        file.write("condition, confidence, time, runid" + '\n')
        for run in all_confidences:
            for t, d in run.items():
                rid = d[0]
                c = d[1]
                file.write('{},{},{},{}\n'.format('easy->hard', c, t, rid))

    # hard -> easy -> hard
    zones, craters, deliveries, assessments, confidence, all_confidences = run_difficulty(num_runs,
                                                                         craters1, craters_empty, craters2,
                                                                         zones1, zones_empty, zones2)
    conditions = []
    conf = []
    time = []
    runid = []
    idx = 0
    confidence = confidence[:100]
    for _run in confidence:
        idx+=1
        for event in _run:
            conditions.append('hard->easy')
            time.append(event[0])
            conf.append(event[1])
            runid.append(idx)
    with open('difficulty_all.csv', 'a') as file:
        for (cond, c, t, rid) in zip(conditions, conf, time, runid):
            file.write('{},{},{}\n'.format(cond, c, t, rid))

    with open('difficulty_all_conf.csv', 'a') as file:
        for run in all_confidences:
            for t, d in run.items():
                rid = d[0]
                c = d[1]
                file.write('{},{},{},{}\n'.format('hard->easy', c, t, rid))


def random_guided_human():
    num_runs = 50
    craters_run = []
    zones_run = []
    event_craters = []
    event_zones = []
    for i in range(num_runs):
        craters1, zones1 = configs.create_new_craters(size=100)
        craters_run.append(craters1)
        zones_run.append(zones1)
        craters2, zones2 = configs.create_new_craters(size=100)
        event_craters.append(craters2)
        event_zones.append(zones2)

    # human_zones, human_craters, human_deliveries = run('human', num_runs, craters_run, event_craters)
    random_zones, random_craters, random_deliveries = run('random', num_runs, craters_run, event_craters, zones_run,
                                                          event_zones)
    static_zones, static_craters, static_deliveries = run('static', num_runs, craters_run, event_craters, zones_run,
                                                          event_zones)

    fig, (a1, a2) = plt.subplots(nrows=1, ncols=2)

    a1.boxplot([random_craters, static_craters])  # , human_deliveries])
    a1.set_xticks([1, 2], ['random choice', 'confidence guided'])  # 'human+confidence guided'])
    a1.axhline(y=5, c='red', linestyle='dashed')
    a1.set_ylabel('Crater hits')
    a1.set_xlabel('Condition')
    a1.set_title('Crater hits per task')

    a2.bar(['random choice', 'confidence guided'], [random_deliveries, static_deliveries], width=0.1, edgecolor='black',
           color='white')
    a2.set_ylabel('Deliveries')
    a2.set_xlabel('Condition')
    table = [[random_deliveries, num_runs - random_deliveries], [static_deliveries, num_runs - static_deliveries]]
    oddsratio, pvalue = fisher_exact(table, alternative='less')
    # chisq, p = chisquare(f_obs=[random_deliveries, static_deliveries],f_exp=[])
    a2.set_title('Total deliveries (p={:.3f}, FET)'.format(pvalue))

    plt.tight_layout()
    plt.savefig('plot.png', dpi=400)
    plt.show()


def random_guided_dynamic():
    num_runs = 100
    craters_run = []
    zones_run = []
    event_craters = []
    event_zones = []
    gl = [(45, 20), (14, 40), (38, 47)]
    for i in range(num_runs):
        choices = [0, 1, 2]
        l = np.random.choice(a=choices)
        lxrange = (gl[l][0] - 7, gl[l][0] + 7)  # (5, 45)
        lyrange = (gl[l][1] - 7, gl[l][1] + 7)  # (5, 45)
        srange = (2, 7)

        craters11, zones11 = configs.create_new_craters(30,
                                                        lxrange=lxrange,
                                                        lyrange=lyrange,
                                                        sxrange=srange,
                                                        syrange=srange)

        l = np.random.choice(a=choices)
        lxrange = (gl[l][0] - 7, gl[l][0] + 7)
        lyrange = (gl[l][1] - 7, gl[l][1] + 7)
        srange = (2, 7)

        craters12, zones12 = configs.create_new_craters(30,
                                                        lxrange=lxrange,
                                                        lyrange=lyrange,
                                                        sxrange=srange,
                                                        syrange=srange)
        craters_run.append(craters11 + craters12)
        zones_run.append(zones11 + zones12)

        choices.remove(l)
        l = np.random.choice(a=choices)
        lxrange = (gl[l][0] - 5, gl[l][0] + 5)  # (5, 45)
        lyrange = (gl[l][1] - 5, gl[l][1] + 5)  # (5, 45)
        srange = (2, 7)
        craters21, zones21 = configs.create_new_craters(25,
                                                        lxrange=lxrange,
                                                        lyrange=lyrange,
                                                        sxrange=srange,
                                                        syrange=srange)

        l = np.random.choice(a=choices)
        lxrange = (gl[l][0] - 5, gl[l][0] + 5)  # (5, 45)
        lyrange = (gl[l][1] - 5, gl[l][1] + 5)  # (5, 45)
        srange = (2, 7)
        craters22, zones22 = configs.create_new_craters(25,
                                                        lxrange=lxrange,
                                                        lyrange=lyrange,
                                                        sxrange=srange,
                                                        syrange=srange)
        event_craters.append(craters21 + craters22)
        event_zones.append(zones21 + zones22)

    labels = ['random', 'static', 'triggered']
    for l in labels:
        zones, craters, deliveries, assessments = run(l, num_runs,
                                                      craters_run, event_craters,
                                                      zones_run, event_zones)
        print("{} deliveries: {}".format(l, sum(deliveries)))
        write_csv(l, zones, craters, deliveries, assessments)

    labels = ['dynamic-random', 'dynamic-static', 'dynamic-triggered']
    for l in labels:
        zones, craters, deliveries, assessments = run(l, num_runs,
                                                      craters_run, event_craters,
                                                      zones_run, event_zones)
        print("{} deliveries: {}".format(l, sum(deliveries)))
        write_csv(l, zones, craters, deliveries, assessments)


def make_dynamic():
    random = pd.read_csv("dynamic-random.csv")
    static = pd.read_csv("dynamic-static.csv")
    triggered = pd.read_csv("dynamic-triggered.csv")
    data = pd.concat([random, static, triggered])
    data.to_csv('all_data_dynamic.csv', index=False)


def make_static():
    random = pd.read_csv("random.csv")
    static = pd.read_csv("static.csv")
    triggered = pd.read_csv("triggered.csv")
    data = pd.concat([random, static, triggered])
    data.to_csv('all_data_static.csv', index=False)

def make_all():
    static = pd.read_csv('all_data_static.csv')
    dynamic = pd.read_csv('all_data_dynamic.csv')
    data = pd.concat([static, dynamic])
    data.to_csv('data_all.csv', index=False)

if __name__ == '__main__':
    random_guided_dynamic()
    #make_static()
    #make_dynamic()
    #make_all()
    task_difficulty()

