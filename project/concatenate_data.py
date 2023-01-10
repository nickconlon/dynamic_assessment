import pandas as pd


def make_dynamic():
    random = pd.read_csv("dynamic-random.csv")
    static = pd.read_csv("dynamic-static.csv")
    triggered = pd.read_csv("dynamic-triggered.csv")

    random['environment'] = ['dynamic'] * len(random)
    random['condition'] = ['random'] * len(random)
    static['environment'] = ['dynamic'] * len(random)
    static['condition'] = ['static'] * len(random)
    triggered['environment'] = ['dynamic'] * len(random)
    triggered['condition'] = ['dynamic'] * len(random)

    data = pd.concat([random, static, triggered])
    data.to_csv('all_data_dynamic.csv', index=False)

def make_all():
    all_dynamic = pd.read_csv("all_data_dynamic.csv")
    all_static = pd.read_csv("all_data_static.csv")

    data = pd.concat([all_static, all_dynamic])
    data.to_csv('all_data.csv', index=False)


def make_static():
    random = pd.read_csv("random.csv")
    static = pd.read_csv("static.csv")
    triggered = pd.read_csv("triggered.csv")

    random['environment'] = ['static'] * len(random)
    static['environment'] = ['static'] * len(random)
    triggered['environment'] = ['static'] * len(random)
    triggered['condition'] = ['dynamic'] * len(random)

    data = pd.concat([random, static, triggered])
    data.to_csv('all_data_static.csv', index=False)


make_static()
make_dynamic()
make_all()