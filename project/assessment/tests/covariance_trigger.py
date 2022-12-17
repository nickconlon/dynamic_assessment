import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats

from project.solvers.q_learning_policy import q_policy
import project.configs as configs
import project.environment as Env
from project.assessment.assessments import StaticAssessment
from triggers_test import empirical_transition_function

def confidence_ellipse(x, y, ax, n_std=3.0, color='red', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      color=color, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


T, S0 = empirical_transition_function()

s = configs.LOCATIONS[configs.HOME].position
g = configs.LOCATIONS[configs.AREA_1].position

env = Env.Environment(s, g)
policy = q_policy(configs.LOCATIONS[configs.AREA_1].policy)

xs = []
si_actual = []
si_ideal = []
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

##############
si_rollout = []
si_rollout_threshold = 0.1
si_mp = []
si_mp_threshold = 0.1
cov = []
cov_threshold = 3
thresh = []
thresh_threshold = 3
################


def threshold_trigger(_actual, _predicted, _threshold):
    return 1 if abs(_actual - np.mean(_predicted)) > _threshold else 0


def covariance_trigger(_actual, _prediction):
    dims = len(_actual)
    trigger = 0
    for d in range(dims):
        _act = _actual[d]
        _mu = np.mean(_prediction[:, d])
        _two_sig = 2 * np.std(_prediction[:, d])
        if abs(_mu - _act) > _two_sig:
            return 1
    return 0


def surprise_index_trigger(_actual, _predicted, ax):
    # TODO Analytically determine the surprise & compare it to the modeled surprise
    mu = np.mean(_predicted)
    sig = np.std(_predicted)
    x = stats.norm(mu, sig)
    d = x.pdf(_actual)
    print(x.cdf(_))
    plt.hist(d)
    plt.show()
    '''
    dpredicted = _predicted[~np.isnan(_predicted)]
    if len(dpredicted) == 0:
        return 0.0
    if len(np.unique(dpredicted)) == 1:
        dpredicted = np.array([dpredicted[0] - 1, dpredicted[0], dpredicted[0] + 1])
    kernel = stats.gaussian_kde(dpredicted, bw_method='scott')
    xx = np.linspace(min(dpredicted) - 25, max(dpredicted) + 25, 500)
    smaller = np.where(kernel(xx) < kernel(_actual))
    p_actual = kernel(_actual)[0]
    p_distribution = kernel(xx)
    p_distribution_smaller = p_distribution[smaller]
    ax.clear()
    surprise = np.trapz(p_distribution_smaller, xx[smaller])
    ax.plot(xx, p_distribution)
    ax.hist(dpredicted, density=True)
    ax.plot([_actual, _actual], [0, p_actual], color='red', linewidth=3)
    ax.plot(xx[smaller], p_distribution_smaller)
    ax.set_title('SI={:.2f}'.format(surprise))
    '''
    return 0#surprise

zones = []#Env.Obstacle(int(x), int(y), 3, configs.ZONE_COLOR) for (x, y) in
         #zip(np.random.normal(loc=20, scale=5, size=100), np.random.normal(loc=20, scale=5, size=100))]
craters = []
env.change_event(new_zones=zones, new_craters=craters)
sa = StaticAssessment()
rewards, collisions, predictions, times = sa.rollout(policy, env, s, 20, 0.5)
print("Reward: ", np.mean(rewards))
prediction_index = 2

TT = copy.deepcopy(T)
S = copy.deepcopy(S0)

env.change_event(new_zones=[], new_craters=[])
env.reset(state=s)
for i in range(60):
    s_hat = np.dot(S, TT)
    a = policy.noisy_pi(s, p_correct=1)
    #if 20 < i < 25:
    #    a = 0
    # if 40 < i < 45:
    #    a = 0
    s, _, done, _ = env.step(a)

    predictions_x = copy.deepcopy(predictions[:, prediction_index, 0])
    predictions_y = copy.deepcopy(predictions[:, prediction_index, 1])

    mu_x.append(np.mean(predictions_x))
    #sigma_x.append(np.std(predictions_x))
    position_actual_x.append(env.xy_from_index(s)[0])
    mu_y.append(np.mean(predictions_y))
    #sigma_y.append(np.std(predictions_y))
    position_actual_y.append(env.xy_from_index(s)[1])

    #############
    si_mp.append(np.sum(s_hat[np.where(s_hat < s_hat[s])]))
    si_rollout.append(surprise_index_trigger(env.xy_from_index(s)[0], predictions[:, prediction_index, 0], a4))
    cov.append(covariance_trigger(env.xy_from_index(s), predictions[:, prediction_index, :]))
    thresh.append(threshold_trigger(env.xy_from_index(s)[0], predictions_x, thresh_threshold))
    #############

    if covariance_trigger(env.xy_from_index(s), predictions[:, prediction_index, :]) == 1:
        print("rollout")
        rewards, collisions, predictions, times = sa.rollout(policy, copy.deepcopy(env),
                                                             copy.deepcopy(env.xy_from_index(s)), 20, 0.95)
        print("Reward: ", -i+np.mean(rewards))
        prediction_index = 0
        S[:] = 0
        S[s] = 1.0
    prediction_index += 1
    xs.append(i)

    a1.clear()
    a2.clear()
    #a3.clear()
    confidence_ellipse(predictions_x, 50-predictions_y, a3, n_std=2, color='red', alpha=0.2)

    a3.plot(mu_x, [50-y for y in mu_y], color='black')
    a3.plot(position_actual_x, [50-y for y in position_actual_y], color='blue')

    a1.imshow(env.render(mode='rgb_array'))

    a2.plot(xs, si_rollout, color='blue')
    a2.plot(xs, cov, color='black')
    a2.plot(xs, si_mp, color='orange')

    a3.set_xlim([0, 50])
    a3.set_ylim([0, 50])
    a2.set_ylim([-0.1, 1.1])
    a2.set_xlim([0, 50])
    plt.pause(0.01)

    TT = np.dot(TT, T)

    if done:
        break

plt.show()
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

