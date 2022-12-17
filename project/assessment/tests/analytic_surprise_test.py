import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats
from scipy.linalg import sqrtm


def plot_ellipse(w, v, ax):
    origin = [0, 0]
    e1 = v[:, 0]
    e2 = v[:, 1]
    ax.quiver(*origin, *e1, color='blue', scale=1 / w[0], scale_units='xy')
    ax.quiver(*origin, *e2, color='red', scale=1 / w[1], scale_units='xy')
    theta = np.linspace(0, 2 * np.pi, 1000)
    ellipsis = ((w[None, :]) * v) @ [np.sin(theta), np.cos(theta)]
    ax.plot(ellipsis[0, :], ellipsis[1, :], color='red')


def rotate(pred, theta_deg):
    theta = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    for i in range(len(pred)):
        pred[i,:] = np.dot(rot, pred[i,:])
    return pred


def fast_check(_predicted, _actual, threshold):
    """

    :param _predicted:  Numpy array of the form [num_measurements, num_features]
    :param _actual:     Numpy array of the form [num_features]
    :param threshold:   Sigma threshold for considering an _actual measurement an anomaly
    :return:            True if anomaly, false otherwise
    """
    #########
    # shift to (0,0)
    mu = np.mean(_predicted, axis=0)
    cov = np.cov(_predicted.T)
    _predicted -= mu
    _actual -= mu

    w, v = np.linalg.eig(cov)

    #########
    # rotate the data so its along an axis TODO this is still a bit weird
    rad_tan = np.arctan2(v[0, 1], v[0, 0])
    deg_tan = np.rad2deg(rad_tan)

    pred = rotate(_predicted, 180 + deg_tan)
    _actual = rotate(np.asarray([_actual]), 180 + deg_tan)[0]
    cov = np.cov(pred.T)
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


def eigin(pred, actual, threshold, a1, a2):
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
    rad_tan = np.arctan2(v[0,1], v[0,0])
    deg_tan = np.rad2deg(rad_tan)
    if abs(rad_tan-0)%360 < 5 or abs(rad_tan-90)%360 < 5 or abs(rad_tan-180)%360 < 5 or abs(rad_tan-270)%360 < 5:
        print("Don't Rotate")
        pass
    else:
        print("Rotate")
        #rad_cos = np.arccos(np.dot(v[0,:], np.array([1,0]))/np.linalg.norm(v[0,:]))
        #deg_cos = np.rad2deg(rad_cos)
        print('rot ', deg_tan)
        print('rot ', 180+deg_tan)
        #print(deg_cos)
        #print(180+deg_cos)
        pred = rotate(pred, 180+deg_tan)
        actual = rotate(np.asarray([actual]), 180+deg_tan)[0]
        cov = np.cov(pred.T)
        if np.sum(cov) == 0:
            cov = np.identity(len(cov))
        w, v = np.linalg.eig(cov)
    if plot:
        a1.scatter(pred[:, 0], pred[:, 1], color='black')
        a1.scatter(*actual, color='red')
        a1.set_title("rotated")
        plot_ellipse(w, v, a1)

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
    plot_ellipse(w1, v1, a2)
    #print(cov1)
    #print(w1)
    #print(v1)

    #########
    # detect an outlier at 1-sigma
    r = threshold  # 1 sig
    theta = np.linspace(0, 2 * np.pi, 100)
    a2.plot(r * np.cos(theta), r * np.sin(theta))
    a2.scatter(*actual, color='red')
    a2.set_title("Scaled")
    if np.linalg.norm(actual) > r:
        return "anomoly"


def surprise_index_trigger(_actual, _predicted, ax):
    # TODO Analytically determine the surprise & compare it to the modeled surprise
    mu = np.mean(_predicted)
    sig = np.std(_predicted)
    x = stats.norm(mu, sig)
    d = np.abs((mu - _actual))
    si = x.cdf(-d)

    ax.plot(np.linspace(-10, 10, 100), x.pdf(np.linspace(-10, 10, 100)))
    ax.fill_between(np.linspace(-10, -_actual, 100), 0 + x.pdf(np.linspace(-10, -_actual, 100)), color='red', alpha=0.2)
    ax.fill_between(np.linspace(_actual, 10, 100), 0 + x.pdf(np.linspace(_actual, 10, 100)), color='red', alpha=0.2)

    print(2 * si)
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
    return 0  # surprise


fig, (a1, a2, a3) = plt.subplots(nrows=1, ncols=3)

#cov = np.array([[2, 0], [0, 10]])
cov = np.array([[10, 0], [0, 10]])
pred = np.random.multivariate_normal([0, 0], cov, size=1000)
pred = np.ones((100, 2))*10
actual = np.array([10, 10.])
#pred = rotate(pred, 90)
a1.scatter(pred[:, 0], pred[:, 1])
a1.scatter(*actual, color='red')
plot_ellipse(*np.linalg.eig(cov), a1)

a1.set_xlim([-20, 20])
a2.set_xlim([-20, 20])
a3.set_xlim([-20, 20])
a1.set_ylim([-20, 20])
a2.set_ylim([-20, 20])
a3.set_ylim([-20, 20])
plt.axis('equal')
a = eigin(pred, actual, 2, a2, a3)

#t1 = time.time()
#a = fast_check(pred, actual, 2)
#t2 = time.time()
#print(t2-t1)
#print(a)

plt.show()
