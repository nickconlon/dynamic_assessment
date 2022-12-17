import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def discrete():
    # fname = 'no_noise_discrete.npy'
    fname = 'noise_discrete.npy'

    preds = np.load(fname)
    # preds = preds[:, :44, :]
    preds[preds == 0] = np.nan
    meanx = np.nanmean(preds[:, :, 0], axis=0)
    stdx = np.nanstd(preds[:, :, 0], axis=0)
    meany = np.nanmean(preds[:, :, 1], axis=0)
    stdy = np.nanstd(preds[:, :, 1], axis=0)

    # stdx = np.concatenate((np.asarray([5]*(len(meany)-25)), np.asarray([10]*25)))
    plt.scatter(meanx, meany)
    plt.fill_between(meanx, meany + stdx, meany - stdx, facecolor='blue', alpha=0.3)
    plt.tight_layout()
    plt.show()


def continuous():
    fname = r'C:\DATA\caml_trajecotries\processed\test_light\techpod_state_{}.npy'
    #fname = r'C:\DATA\caml_trajecotries\processed\isr_wind_strong\techpod_state_{}.npy'
    #fname = r'C:\DATA\caml_trajecotries\processed\\payload\techpod_state_{}.npy'
    alls = np.zeros((5, 61, 2))
    for i in range(5):
        preds = np.load(fname.format(i))
        tmp = preds[:, :2]
        alls[i, :, :] = tmp[:, :]

    meanx = np.mean(alls[:, :, 0], axis=0)
    stdx = np.std(alls[:, :, 0], axis=0)
    meany = np.mean(alls[:, :, 1], axis=0)
    stdy = np.std(alls[:, :, 1], axis=0)

    plt.plot(range(len(meanx)), meanx)
    plt.fill_between(range(len(meanx)), meanx + stdx, meanx - stdx, facecolor='blue', alpha=0.3)
    test = 0
    mu = meanx[35]
    std = stdx[35]
    z = (test-mu)/std
    plt.title(stats.norm.sf(abs(z))*2)
    plt.show()
    plt.plot(range(len(meany)), meany)
    plt.fill_between(range(len(meany)), meany + stdy, meany - stdy, facecolor='blue', alpha=0.3)

    plt.show()
    for i in range(5):
        plt.plot(alls[i, :, 0], alls[i, :, 1])
    plt.show()

def SMAPE(F, A):
    # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    return (100/len(F))*np.sum((F-A)/(np.abs(A)+np.abs(F)))

def SURPRISE(F, A):
    pass

def dynamic_time_warping():
    from dtaidistance import dtw
    #fname = r'C:\DATA\caml_trajecotries\processed\real_calm_10\techpod_state_{}.npy'
    #fname = r'C:\DATA\caml_trajecotries\processed\isr_wind_light\techpod_state_{}.npy'
    #fname = r'C:\DATA\caml_trajecotries\processed\isr_wind_strong\techpod_state_{}.npy'
    fname = r'C:\DATA\caml_trajecotries\processed\\payload\techpod_state_{}.npy'
    alls = np.zeros((5, 61, 2))
    for i in range(5):
        preds = np.load(fname.format(i))
        tmp = preds[:, :2]
        alls[i, :, :] = tmp[:, :]

    meanx = np.mean(alls[:, :, 0], axis=0)
    #meanx = alls[1, :, 0]
    for i in range(5):
        candidate = alls[i, :, 0]
        print(dtw.distance_fast(meanx, candidate))
        print(SMAPE(meanx, candidate))


dynamic_time_warping()
