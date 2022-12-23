import copy
import sys
import numpy as np
from scipy import stats
from project.assessment.famsec import FaMSeC


class CompetencyAssessmentBase:
    def __init__(self):
        pass

    def approx_assessment(self, trajectories):
        pass

    def full_assessment(self, trajectories):
        pass


class StaticAssessment(CompetencyAssessmentBase):
    def __init__(self):
        super().__init__()
        self.famsec = FaMSeC()

    @staticmethod
    def rollout(policy, env, state, num_rollouts, transition_uncertainty):
        max_length = 250

        rewards = np.zeros(num_rollouts)
        collisions = np.zeros(num_rollouts)
        zones = np.zeros(num_rollouts)
        times = np.zeros(num_rollouts)
        deliveries = np.zeros(num_rollouts)

        predictions = np.zeros((num_rollouts, max_length, 2)) * np.nan
        for i in range(num_rollouts):
            s = env.reset(state=state)
            predictions[i, 0, :] = env.xy_from_index(s)
            for j in range(max_length - 1):
                # a = policy.pi(s)
                action = policy.noisy_pi(s, transition_uncertainty)
                s, reward, done, info = env.step(action)
                predictions[i, j + 1, :] = env.xy_from_index(s)

                collisions[i] += info['collisions']
                zones[i] += info['zones']
                times[i] += 1
                rewards[i] += info['rewards']
                if done:
                    if collisions[i] < 5:
                        deliveries += 1
                    break

        # np.save('./data/noise_discrete.npy', predictions)
        return rewards, collisions, zones, predictions, times

    def run_assessment(self, policy, env, state, num_rollouts, z_star, transition_uncertainty):
        rewards, collisions, zones, states, times = self.rollout(policy, env, state, num_rollouts, transition_uncertainty)
        oa = self.famsec.outcome_assessment(reward_dist=rewards, r_star=z_star)
        return oa, rewards, collisions, times, states

    def run_goa_assessment_new(self, policy, env, state, num_rollouts, z_stars, current_counts, transition_uncertainty):
        rewards, collisions, zones, states, times = self.rollout(policy, env, state, num_rollouts, transition_uncertainty)

        collisions_oa = self.famsec.outcome_assessment(reward_dist=current_counts[1]+collisions, r_star=z_stars[1], swap=True)
        collisions_oa = np.around(collisions_oa, decimals=2)

        zones_oa = self.famsec.outcome_assessment(reward_dist=current_counts[2]+zones, r_star=z_stars[2], swap=True)
        zones_oa = np.around(zones_oa, decimals=2)
        return collisions_oa, np.mean(collisions), np.std(collisions), zones_oa, np.mean(zones), np.std(zones), states

    def run_goa_assessment(self, policy, env, state, num_rollouts, z_stars, current_counts, transition_uncertainty) -> object:
        rewards, collisions, zones, states, times = self.rollout(policy, env, state, num_rollouts, transition_uncertainty)

        rewards_oa = self.famsec.outcome_assessment(reward_dist=current_counts[0]+rewards, r_star=z_stars[0])
        rewards_oa = np.around(rewards_oa, decimals=2)
        collisions_oa = self.famsec.outcome_assessment(reward_dist=current_counts[1]+collisions, r_star=z_stars[1], swap=True)
        collisions_oa = np.around(collisions_oa, decimals=2)
        #times_oa = self.famsec.outcome_assessment(reward_dist=current_counts[2]+times, r_star=z_stars[2], swap=True)
        #times_oa = np.around(times_oa, decimals=2)
        zones_oa = self.famsec.outcome_assessment(reward_dist=current_counts[2]+zones, r_star=z_stars[2], swap=True)
        zones_oa = np.around(zones_oa, decimals=2)
        return rewards_oa, collisions_oa, zones_oa, states


class DynamicAssessment(CompetencyAssessmentBase):
    def __init__(self):
        super().__init__()
        self.metric = 1

    @staticmethod
    def assessment(actual, dpredicted, ax=None):
        dpredicted = dpredicted[~np.isnan(dpredicted)]
        if len(dpredicted) == 0:
            return 0.0
        if len(np.unique(dpredicted)) == 1:
            dpredicted = np.array([dpredicted[0] - 5, dpredicted[0], dpredicted[0] + 5])
        kernel = stats.gaussian_kde(dpredicted, bw_method='scott')
        xx = np.linspace(min(dpredicted) - 25, max(dpredicted) + 25, 500)
        smaller = np.where(kernel(xx) < kernel(actual))
        p_actual = kernel(actual)[0]
        p_distribution = kernel(xx)
        p_distribution_smaller = p_distribution[smaller]

        surprise = np.trapz(p_distribution_smaller, xx[smaller])
        if ax is not None:
            ax.clear()
            ax.plot(xx, p_distribution)
            ax.hist(dpredicted, density=True)
            ax.plot([actual, actual], [0, p_actual], color='red', linewidth=3)
            ax.plot(xx[smaller], p_distribution_smaller)
            ax.set_title('SI={:.2f}'.format(surprise))
        return surprise

    @staticmethod
    def rotate(pred, theta_deg):
        theta = np.deg2rad(theta_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        for i in range(len(pred)):
            pred[i, :] = np.dot(rot, pred[i, :])
        return pred

    @staticmethod
    def plot_ellipse(w, v, ax):
        origin = [0, 0]
        e1 = v[:, 0]
        e2 = v[:, 1]
        ax.quiver(*origin, *e1, color='blue', scale=1 / w[0], scale_units='xy')
        ax.quiver(*origin, *e2, color='red', scale=1 / w[1], scale_units='xy')
        theta = np.linspace(0, 2 * np.pi, 1000)
        ellipsis = ((w[None, :]) * v) @ [np.sin(theta), np.cos(theta)]
        ax.plot(ellipsis[0, :], ellipsis[1, :], color='red')

    def fast_sigma_bounds(self, _predicted, _actual, threshold):
        """

        :param _predicted:  Numpy array of the form [num_measurements, num_features]
        :param _actual:     Numpy array of the form [num_features]
        :param threshold:   Sigma threshold for considering an _actual measurement an anomaly
        :return:            True if anomaly, false otherwise
        """
        pred = copy.deepcopy(_predicted)
        #########
        # shift to (0,0)
        mu = np.mean(pred, axis=0)
        cov = np.cov(pred.T)
        if np.sum(cov) == 0:
            cov = np.identity(len(cov))
        pred -= mu
        _actual -= mu

        w, v = np.linalg.eig(cov)

        #########
        # rotate the data so its along an axis TODO this is still a bit weird
        rad_tan = np.arctan2(v[0, 1], v[0, 0])
        deg_tan = np.rad2deg(rad_tan)
        if abs(deg_tan - 0) % 360 < 5 or abs(deg_tan - 90) % 360 < 5 or abs(deg_tan - 180) % 360 < 5 or abs(
                deg_tan - 270) % 360 < 5:
            print("Don't Rotate")
            pass
        else:
            pred = self.rotate(pred, 180 + deg_tan)
            _actual = self.rotate(np.asarray([_actual]), 180 + deg_tan)[0]
            cov = np.cov(pred.T)
            if np.sum(cov) == 0:
                cov = np.identity(len(cov))
            w, v = np.linalg.eig(cov)

        #########
        # scale to 1 by the eigenvalues
        pred *= 1 / np.sqrt(w)
        _actual *= 1 / np.sqrt(w)

        #########
        # detect an outlier at 1-sigma
        r = threshold  # 1 sig

        if np.linalg.norm(_actual) > r:
            return "Anomaly"
        else:
            return "No anomaly"

    def eigin(self, pred, actual, threshold, a1, a2):
        plot = True
        #########
        # shift to (0,0)
        cov = np.cov(pred.T)
        mu = np.mean(pred, axis=0)
        if np.sum(cov) == 0:
            cov = np.identity(len(cov))
        pred -= mu
        actual -= mu

        w, v = np.linalg.eig(cov)

        #########
        # rotate by eigenvalues TODO this is still a bit weird
        rad_tan = np.arctan2(v[0, 1], v[0, 0])
        deg_tan = np.rad2deg(rad_tan)
        if abs(rad_tan - 0) % 360 < 5 or abs(rad_tan - 90) % 360 < 5 or abs(rad_tan - 180) % 360 < 5 or abs(
                rad_tan - 270) % 360 < 5:
            print("Don't Rotate")
            pass
        else:
            print("Rotate")
            # rad_cos = np.arccos(np.dot(v[0,:], np.array([1,0]))/np.linalg.norm(v[0,:]))
            # deg_cos = np.rad2deg(rad_cos)
            print('rot ', deg_tan)
            print('rot ', 180 + deg_tan)
            # print(deg_cos)
            # print(180+deg_cos)
            pred = self.rotate(pred, 180 + deg_tan)
            actual = self.rotate(np.asarray([actual]), 180 + deg_tan)[0]
            cov = np.cov(pred.T)
            if np.sum(cov) == 0:
                cov = np.identity(len(cov))
            w, v = np.linalg.eig(cov)
        if plot:
            a1.scatter(pred[:, 0], pred[:, 1], color='black')
            a1.scatter(*actual, color='red')
            a1.set_title("rotated")
            self.plot_ellipse(w, v, a1)

        #########
        # scale by eigenvalues
        pred *= 1 / np.sqrt(w)
        actual *= 1 / np.sqrt(w)

        #########
        # plot the updated covariance estimate
        cov1 = np.cov(pred.T)
        if np.sum(cov1) == 0:
            cov1 = np.identity(len(cov))
        w1, v1 = np.linalg.eig(cov1)
        a2.scatter(pred[:, 0], pred[:, 1], color='black')
        self.plot_ellipse(w1, v1, a2)
        # print(cov1)
        # print(w1)
        # print(v1)

        #########
        # detect an outlier at 1-sigma
        r = threshold  # 1 sig
        theta = np.linspace(0, 2 * np.pi, 100)
        a2.plot(r * np.cos(theta), r * np.sin(theta))
        a2.scatter(*actual, color='red')
        a2.set_title("Scaled")
        if np.linalg.norm(actual) > r:
            return "anomoly"

    @staticmethod
    def distance(pred, actual, threshold):
        _mu = np.mean(pred, axis=0)
        d = np.linalg.norm(_mu - actual)
        return d <= threshold, d

    def conf_ellipse(self, pred, actual, nstd):
        # https://stackoverflow.com/questions/66010530/how-to-find-point-inside-an-ellipse
        pass

    @staticmethod
    def tail(pred, actual):
        # Assuming this is from a multivariate Gaussian distribution...
        # https://stats.stackexchange.com/questions/331283/how-to-calculate-the-probability-of-a-data-point-belonging-to-a-multivariate-nor
        # https: // www.visiondummy.com / 2014 / 04 / draw - error - ellipse - representing - covariance - matrix /
        # https://stats.stackexchange.com/questions/105133/tail-probabilities-of-multivariate-normal-distribution
        if pred.shape[0] <= 1:
            sigma = np.array([[2., 0], [0, 2]])
            print('manual sigma')
        else:
            sigma = np.cov(pred.T)

        x = copy.deepcopy(actual)
        mu = np.mean(pred, axis=0)
        try:
            if np.linalg.cond(sigma) < 1 / sys.float_info.epsilon:
                pass
            else:
                sigma = np.array([[1., 0], [0, 1]])
        except Exception as e:
            print(e)
            print('\nsigma \n', sigma)
            print('\npred \n', pred)
            print('\n')

        m_dist_x = np.dot((x - mu).transpose(), np.linalg.inv(sigma))
        m_dist_x = np.dot(m_dist_x, (x - mu))

        return 1 - stats.chi2.cdf(m_dist_x, len(actual))
'''
d = DynamicAssessment()
pred = np.ones((10,2))*np.array([10,11])
actual = np.array([10, 11])
d.tail(pred, actual)

plt.scatter(pred[:, 0], pred[:, 1], color='blue')
plt.scatter(*actual, color='red')
plt.axis('equal')
plt.show()
####
m = np.array([[1,0],[0,0]], dtype=float)
zeros = np.where(np.diag(m) == 0)
for i in zeros:
    m[i,i] = 99
print(m)
'''