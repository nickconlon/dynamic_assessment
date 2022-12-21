import glob
import json
import project.multiagent_configs as configs
import re

'''
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
'''


def get_rewards_deliveries(infile):
    rewards = 0
    deliveries = 0
    zones = 0
    collisions = 0
    with open(infile) as file:
        for line in file:
            line = line.strip()
            l = re.split('AM | PM ', line)
            if len(l) == 2:
                ll = json.loads(l[1])
                if ll['topic'] == configs.MessageHelpers.TOPICS_STATE_UPDATE:
                    d = ll['data']
                    rewards = d[configs.MultiAgentState.STATUS_REWARDS]
                    deliveries = d[configs.MultiAgentState.STATUS_DELIVERIES]
                    zones = d[configs.MultiAgentState.STATUS_ZONES]
                    collisions = d[configs.MultiAgentState.STATUS_CRATERS]
    return rewards, deliveries, zones, collisions


def get_trust(infile):
    return 'n/a'


def get_workload(infile):
    return 'n/a'


def get_data_from_file(infile):
    metadata = infile.split('.')[0]
    metadata = metadata.split('_')
    rewards, deliveries, zones, collisions = get_rewards_deliveries(infile)
    agent_id = metadata[0]
    condition_id = metadata[1]
    subject_id = metadata[2]
    workload = get_workload('')
    trust = get_trust('')

    return agent_id, condition_id, subject_id, rewards, deliveries, zones, collisions, workload, trust


def convert_to_csv(infiles, outfile):
    header = 'agent id, condition id, subject id, rewards, deliveries, zones, collisions, workload, trust'

    with open(outfile, 'w') as file:
        file.write(header + '\n')
        for infile in infiles:
            agent_id, condition_id, subject_id, rewards, deliveries, zones, collisions, workload, trust = get_data_from_file(infile)
            file.write('{},{},{},{},{},{},{},{},{}\n'.format(agent_id, condition_id, subject_id,
                                                             rewards, deliveries, zones, collisions,
                                                             workload, trust))


fname_logs = glob.glob('*.log')
# fname_logs = '1_1_1001.log', '2_1_1001.log'
fname_csv = 'out.csv'
convert_to_csv(fname_logs, fname_csv)
