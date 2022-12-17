import numpy as np
import matplotlib.pyplot as plt
import copy

import project.configs as configs
import project.environment as Env
from project.solvers.q_learning_policy import q_policy
from project.assessment.assessments import StaticAssessment

initial_state = configs.HOME
goal_state = configs.AREA_3


def empirical_transition_function():
    s0 = configs.LOCATIONS[initial_state].position
    g = configs.LOCATIONS[goal_state].position
    env = Env.Environment(s0, g)
    gidx = env.index_from_xy(*g)
    s0idx = env.index_from_xy(*s0)

    policy = q_policy(configs.LOCATIONS[goal_state].policy)
    T = np.zeros((2500, 2500))

    for x in range(50):
        for y in range(50):
            s = np.asarray([x, y])
            best_a = policy.pi(env.index_from_xy(*s))
            for a in range(4):
                env.reset(state=s)
                sp, _, _, _ = env.step(a)
                if a == best_a:
                    T[env.index_from_xy(*s), sp] = 0.9
                else:
                    T[env.index_from_xy(*s), sp] = (1 - 0.9) / 3

    T[gidx, :] = 0.0
    T[gidx, gidx] = 1.0

    S0 = np.zeros_like(T[:, 0], dtype=float)
    S0[s0idx] = 1.0
    return T, S0


def covariance_trigger(_actual, _predicted, threshold):
    _mu = np.mean(_predicted)
    _two_sig = threshold * np.std(_predicted)
    return 1 if abs(_mu - _actual) > _two_sig else 0


def threshold_trigger(_actual, _predicted, _threshold):
    return 1 if abs(_actual - np.mean(_predicted)) > _threshold else 0


def rollout_surprise_trigger():
    return 0


def markov_process_surprise_trigger():
    return 0


if __name__=='__main__':
    s = configs.LOCATIONS[initial_state].position
    g = configs.LOCATIONS[goal_state].position

    T, s0 = empirical_transition_function()

    fig, (a1, a2, a3) = plt.subplots(nrows=1, ncols=3)

    env = Env.Environment(s, g)
    policy = q_policy(configs.LOCATIONS[goal_state].policy)

    zones = []
    craters = []
    env.change_event(new_zones=zones, new_craters=craters)
    sa = StaticAssessment()
    rewards, collisions, predictions, times = sa.rollout(policy, env, s, 20, 0.8)
    prediction_index = 2

    env.change_event(new_zones=[], new_craters=[])
    env.reset(state=s)

    xs = []
    si_rollout = []
    si_rollout_threshold = 0.1
    si_mp = []
    si_mp_threshold = 0.1
    cov = []
    cov_threshold = 3
    thresh = []
    thresh_threshold = 3

    s_idx = env.index_from_xy(*s)
    for i in range(60):

        a = policy.noisy_pi(s_idx, p_correct=1)
        env.render()
        # if 20 < i < 25:
        #    a = 0
        # if 40 < i < 45:
        #    a = 0
        s_idx, _, done, _ = env.step(a)
        s_actual = env.xy_from_index(s_idx)
        predictions_x = predictions[:, prediction_index, 0]
        predictions_y = predictions[:, prediction_index, 1]
        xs.append(i)

        cov.append(covariance_trigger(s_actual[0], predictions_x, cov_threshold))
        thresh.append(threshold_trigger(s_actual[0], predictions_x, thresh_threshold))
        si_rollout.append(rollout_surprise_trigger())
        si_mp.append(markov_process_surprise_trigger())

        if done:
            break

    plt.xlim(0, 100)
    plt.ylim(-2, 2)
    plt.plot(xs, cov, color='blue')
    plt.show()
