import numpy as np
import matplotlib.pyplot as plt


def bayes(p_o_given_m, p_m):
    idxs = len(p_o_given_m)
    normalizer = 0

    p_m_give_o = np.zeros(idxs)
    for i in range(idxs):
        normalizer += p_o_given_m[i] * p_m[i]

    print("N, ", normalizer)
    for i in range(idxs):
        p_m_give_o[i] = p_o_given_m[i] * p_m[i] / normalizer
    return p_m_give_o


def run_bayes():
    # p(A|X)=p(X|A)p(A) / p(X|A)p(A)+p(X|~A)P(~A)
    # H1 = following the model
    # H2 = not following the model
    priors = np.array([0.5, 0.5])  # prior [M1, M2,...]
    likelihoods = np.array([0.9, 0.2])  # likelihood [M1, M2...]
    for i in range(10):
        posteriors = bayes(p_o_given_m=likelihoods, p_m=priors)  # posterior

        print("M1:")
        print(" prior @ t={} : {}".format(i, priors[0]))
        print(" likelihood @ t={} : {}".format(i, likelihoods[0]))
        print(" posterior  @ t={} : {}".format(i, posteriors[0]))

        print("M2:")
        print(" prior @ t={} : {}".format(i, priors[1]))
        print(" likelihood @ t={} : {}".format(i, likelihoods[1]))
        print(" posterior  @ t={} : {}".format(i, posteriors[1]))

        plt.scatter(i, np.log(priors[0] / posteriors[0]), color='blue')
        plt.scatter(i, np.log(priors[1] / posteriors[1]), color='orange')
        plt.ylim([-2, 2])
        plt.xlim([0, 10])
        plt.pause(0.1)

        priors = posteriors
        likelihoods = np.array([likelihoods[0], likelihoods[1]])
        likelihoods = likelihoods / np.sum(likelihoods)
    plt.show()


if __name__ == '__main__':
    run_bayes()
