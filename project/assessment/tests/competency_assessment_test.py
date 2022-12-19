import copy

import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import traceback

from project.assessment.assessments import StaticAssessment, DynamicAssessment
from project.environment import Environment
from project.multiagent_configs import Obstacle
import project.configs as configs
from project.solvers.q_learning_policy import q_policy


class Writer:
    def __init__(self):
        self.run_id = 0
        self.state = []
        self.outcomes = {'collisions': 0, 'rewards': 0, 'times': 0}
        self.assessments = []
        self.rollouts = []

    def next_run(self):
        pass

    def write_run(self):
        id = self.run_id
        fname = '{}_{}:{}.npy'
        ctr = 0
        for s in self.state:  # assume one state array here
            np.save(fname.format('state', ctr, str(id)), s)
            ctr += 1
        ctr = 0
        for k, v in self.outcomes.items():
            pass
            ctr += 1
        ctr = 0
        for a in self.assessments:
            pass
            ctr += 1
        ctr = 0
        for r in self.rollouts:
            np.save(fname.format('rollout', str(id)), r)
            ctr += 1


def density_plot(actual, predictions):
    import seaborn as sns
    plt.clf()
    sns.set_style('white')
    xs = predictions[:, 0]
    xs[xs == 0] = configs.GOAL_STATE[0]
    ys = predictions[:, 1]
    ys[ys == 0] = configs.GOAL_STATE[1]

    sns.kdeplot(x=predictions[:, 0],
                y=predictions[:, 1],
                cmap="Reds",
                shade=True,
                bw_adjust=1)
    plt.scatter(*actual)
    plt.ylim([250, 0])
    plt.xlim([0, 500])
    plt.pause(0.1)


def dist(actual, predictions, t):
    xs = predictions[:, 0]
    xs[xs == 0] = configs.GOAL_STATE[0]
    ys = predictions[:, 1]
    ys[ys == 0] = configs.GOAL_STATE[1]

    py = np.mean(ys, axis=0)
    px = np.mean(xs, axis=0)
    dt = np.linalg.norm(actual - np.array([px, py]))
    plt.scatter(t, dt)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.pause(0.1)


if __name__ == '__main__':
    initial_state = configs.LOCATIONS[configs.HOME].position
    goal_location = configs.LOCATIONS[configs.AREA_1].position
    obstructions = []
    zones = []
    craters = []
    env = Environment(initial_state, goal_location, obstructions, _zones=zones, _craters=craters)
    policy = q_policy('../../'+configs.POLICY)

    sa = StaticAssessment()
    da = DynamicAssessment()
    uncertainty = 0.99
    rewards, collisions, predicted_states, times = sa.rollout(policy,
                                                              copy.deepcopy(env),
                                                              initial_state,
                                                              configs.OA_ROLLOUTS,
                                                              uncertainty)
    r_star = -50
    z_stars = [-60, 5, 67] # reward, collisions, time
    counts = [-30, 4, 0]
    rewards_oa, collision_oa, times_oa = sa.run_goa_assessment(policy, env, initial_state, 10, z_stars, counts, 1)
    # print(configs.convert_famsec(rewards_oa))
    # print(configs.convert_famsec(collision_oa))
    # print(configs.convert_famsec(times_oa))
    # real mission...
    real_craters = 0
    real_time = 0
    real_rewards = 0
    assessment_time = 1

    average_surprise = []
    average_probability = []
    average_distance = []
    num_runs = 100
    for i in range(num_runs):
        time_steps = 0
        assessment_time = 1

        # plots
        si_over_time = []
        dist_over_time = []
        probability_over_time = []
        zones = [Obstacle(int(x), int(y), 2, configs.ZONE_COLOR) for (x, y) in
                 zip(np.random.normal(loc=30, scale=1, size=10),
                     np.random.normal(loc=20, scale=1, size=10))]
        env.change_event(new_zones=[])
        s = env.reset(state=initial_state)
        while True:
            # a = policy.pi(s)
            if time_steps == 25:
                zones = [Obstacle(int(x), int(y), 2, configs.ZONE_COLOR) for (x, y) in
                         zip(np.random.normal(loc=30, scale=1, size=5),
                             np.random.normal(loc=20, scale=1, size=5))]
                env.change_event(new_zones=zones)
            a = policy.noisy_pi(s, uncertainty)
            s, reward, done, info = env.step(a)
            predicted_state_t = copy.deepcopy(predicted_states[:, assessment_time])
            predicted_state_t = predicted_state_t[~np.isnan(predicted_state_t).any(axis=1), :]
            if predicted_state_t.shape[0] > 1:
                # Surprise Index
                SIx = da.assessment(env.xy_from_index(s)[0], predicted_state_t[:, 0])
                SIy = da.assessment(env.xy_from_index(s)[1], predicted_state_t[:, 1])
                si_over_time.append(np.mean(np.asarray([SIy, SIx])))

                # Distance Index
                dswitch, dmetric = da.distance(predicted_state_t, env.xy_from_index(s), 3)
                dist_over_time.append(dmetric)

                try:
                    # Standard deviations out
                    mdist = da.tail(predicted_state_t, env.xy_from_index(s))
                    probability_over_time.append(mdist)

                    plt.clf()
                    mu = np.mean(predicted_state_t, axis=0)
                    plt.scatter(predicted_state_t[:, 0], predicted_state_t[:, 1])
                    plt.scatter(*env.xy_from_index(s))
                    plt.scatter(*mu, color='red')
                    plt.xlim([0, 50])
                    plt.ylim([0, 50])
                    plt.pause(0.1)

                    time_steps += 1
                except Exception as e:
                    print("#################################################")
                    print(e)
                    print(traceback.print_exc())
                    print(predicted_state_t)
                    print(env.xy_from_index(s))
                    print("#################################################")

            real_craters += info['collisions']
            real_time += 1
            real_rewards += reward
            env.render()
            assessment_time += 1

            if done:
                break

        asi = np.mean(np.asarray(si_over_time))
        average_surprise.append(asi)
        ap = np.mean(np.asarray(probability_over_time))
        average_probability.append(ap)
        ad = np.mean(np.asarray(dist_over_time))
        average_distance.append(ad)
    #fig, a1 = plt.subplots(nrows=1, ncols=1)
    #data = [average_surprise, average_probability]
    #labels = ['Averaged (x,y)\nSurprise', 'Average Tail\nProbability']
    #a1.boxplot(data, patch_artist=True)
    #a1.set_xticks([1, 2], labels)
    #a1.set_ylim([0, 1.01])

        plt.show()
        fig, (a1, a2, a3) = plt.subplots(nrows=1, ncols=3)
        a1.plot(np.arange(0, len(si_over_time)), si_over_time, color='blue')
        a1.set_xlim([0, time_steps])
        a1.set_ylim([-0.1, 1.1])
        a1.set_title('Surprise Index [x]')

        a2.plot(np.arange(0, len(dist_over_time)), dist_over_time, color='blue')
        a2.set_xlim([0, time_steps])
        a2.set_ylim([-10, 10])
        a2.set_title('Absolute Distance')

        a3.plot(np.arange(0, len(probability_over_time)), probability_over_time, color='blue')
        a3.set_xlim([0, time_steps])
        a3.set_ylim([-0.1, 1.1])
        a3.set_title('Tail Probability')
        plt.show()


    plt.show()

    print(craters)
