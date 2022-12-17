import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from project.solvers.q_learning_policy import q_policy
import project.configs as configs
import project.environment as Env
from project.assessment.assessments import StaticAssessment


s0 = configs.LOCATIONS[configs.HOME].position

g = configs.LOCATIONS[configs.AREA_1].position

env = Env.Environment(s0, g)
gidx = env.index_from_xy(*g)
s0idx = env.index_from_xy(*s0)

policy = q_policy(configs.LOCATIONS[configs.AREA_1].policy)
T = np.zeros((2500, 2500))

actions = env.actions
for x in range(50):
    for y in range(50):
        s_idx = env.index_from_xy(x, y)
        s = np.asarray([x, y])
        best_a = policy.pi(env.index_from_xy(*s))
        for a in range(4):
            env.reset(state=s)
            sp, _, _, _ = env.step(a)
            if a == best_a:
                T[env.index_from_xy(*s), sp] = 0.99
            else:
                T[env.index_from_xy(*s), sp] = (1-0.99)/3

T[env.index_from_xy(*g), :] = 0.0
T[env.index_from_xy(*g), env.index_from_xy(*g)] = 1.0

S0 = np.zeros_like(T[:, 0], dtype=float)
S0[s0idx] = 1.0

average_surprise = []
for i in range(10):
    print("{}/10".format(i))
    TT = copy.deepcopy(T)
    S = copy.deepcopy(S0)
    s = env.index_from_xy(*s0)

    zones = []
    env.change_event(new_zones=zones)
    env.reset(state=s0)

    xs = []
    si_actual = []
    si_ideal = []
    si_size = []
    entropy = []
    absolute_difference = []
    probability_of_s = []
    mse = []

    mu_x = []
    sigma_x = []
    position_actual_x = []
    two_sig_x = []
    mu_y = []
    sigma_y = []
    position_actual_y = []
    two_sig_y = []

    fig, (a1, a2, a3, a4) = plt.subplots(nrows=1, ncols=4)


    sa = StaticAssessment()
    rewards, collisions, predictions, times = sa.rollout(policy, env, s0, 10, 0.99)
    prediction_index = 2
    zones = [Env.Obstacle(int(x), int(y), 2, configs.ZONE_COLOR) for (x, y) in
                 zip(np.random.normal(loc=30, scale=1, size=10), np.random.normal(loc=20, scale=1, size=10))]
    craters = []
    env.change_event(new_zones=[], new_craters=craters)
    env.reset(state=s0)
    max_steps = 100

    for i in range(max_steps):
        s_hat = np.dot(S, TT)
        a = policy.noisy_pi(s, p_correct=0.99)
        if i == 25:
            zones = [Env.Obstacle(int(x), int(y), 2, configs.ZONE_COLOR) for (x, y) in
                     zip(np.random.normal(loc=30, scale=1, size=5),
                         np.random.normal(loc=20, scale=1, size=5))]
            env.change_event(new_zones=zones)
        #if 40 < i < 45:
        #    a = 0
        s, _, done, _ = env.step(a)
        prob_s = s_hat[s]

        positives = s_hat[np.where(s_hat > 0)]
        entropy.append(-np.sum(positives*np.log2(positives)))
        si_ideal.append(np.sum(s_hat[np.where(s_hat < np.max(s_hat))]))
        si_actual.append(np.sum(s_hat[np.where(s_hat < prob_s)]))
        si_size.append(len(s_hat[np.where(s_hat > 0)]))
        absolute_difference.append(env.xy_from_index(s)[0] - env.xy_from_index(np.argmax(s_hat))[0])
        probability_of_s.append(prob_s)

        predictions_x = predictions[:, prediction_index, 0]
        predictions_y = predictions[:, prediction_index, 1]
        
        mu_x.append(np.mean(predictions_x))
        sigma_x.append(np.std(predictions_x))
        position_actual_x.append(env.xy_from_index(s)[0])
        mu_y.append(np.mean(predictions_y))
        sigma_y.append(np.std(predictions_y))
        position_actual_y.append(env.xy_from_index(s)[1])
        
        #if np.diff(si_actual[-2:]) > 0.1:
        #    print("rollout")
        #    rewards, collisions, predictions, times = sa.rollout(policy, copy.deepcopy(env), copy.deepcopy(env.xy_from_index(s)), 10, 0.9)
        #    prediction_index = 1
        prediction_index += 1
        xs.append(i)

        a1.clear()
        a2.clear()
        a3.clear()
        a1.imshow(env.render(mode='rgb_array'))
        a2.plot(xs, mu_x, color='blue')
        a2.plot(xs, position_actual_x, color='black')
        a2.fill_between(xs, np.asarray(mu_x)-2*np.asarray(sigma_x), np.asarray(mu_x)+2*np.asarray(sigma_x), color='red', alpha=0.2)
        #a3.plot(xs, mu_y, color='blue')
        #a3.plot(xs, position_actual_y, color='black')
        #a3.fill_between(xs, np.asarray(mu_y)-2*np.asarray(sigma_y), np.asarray(mu_y)+2*np.asarray(sigma_y), color='red', alpha=0.2)
        #a2.plot(xs, two_sig, color='red')
        #a3.plot(xs, mse, color='grey')
        #a3.plot(xs, probability_of_s, color='orange')
        #a2.plot(xs, absolute_difference, color='green')
        #a3.plot(xs, entropy, color='black')
        #a2.fill_between(xs, np.asarray(entropy) - np.asarray(entropy_bounds), np.asarray(entropy) + np.asarray(entropy_bounds), color='red',
        #                alpha=0.2)
        a3.plot(xs, si_actual, color='royalblue', label='noisy (actual)')
        a3.plot(xs, si_ideal, color='lightcoral', label='ideal (actual)')
        a4.scatter(si_ideal, si_size, color='red')
        a4.set_xlim([0, 1])
        a4.set_ylim([0, 50*50])
        #a3.plot(xs[:-1], np.diff(si_actual), color='blue', label='noisy (derivative)')
        #a3.plot(xs[:-1], np.diff(si_ideal), color='red', label='ideal (derivative)')
        a2.set_xlim([0, 60])
        a3.set_xlim([0, 60])
        a3.set_ylim([0, 1])
        a2.set_ylim([0, 50])
        plt.pause(0.01)

        if done:
            break
        TT = np.dot(TT, T)
    plt.show()
    average_surprise.append(np.mean(np.asarray(si_actual)))
fig, a1 = plt.subplots(nrows=1, ncols=1)
data = [average_surprise]
labels = ['Averaged (x,y)\nSurprise']
a1.boxplot(data, patch_artist=True)
a1.set_xticks([1], labels)
a1.set_ylim([0, 1.01])
'''
plt.plot(xs, si_actual, color='royalblue', label='noisy (actual)')
plt.plot(xs, si_ideal, color='lightcoral', label='ideal (actual)')
# plt.plot(xs[:-1], np.diff(ys), color='blue', label='noisy (derivative)')
# plt.plot(xs[:-1], np.diff(maxs), color='red', label='ideal (derivative)')

plt.plot([0, 60], [0.01, 0.01], color='green')
plt.plot([0, 60], [-0.01, -0.01], color='green')

plt.xlim([0, 60])
plt.ylim([-1, 1])
plt.ylabel('Surprise')
plt.xlabel('Time')
plt.legend()
'''
plt.show()
