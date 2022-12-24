import numpy as np
import json
import base64
import cv2

#
# Communications and topics
#
HOST = 'localhost'

MAP_TOPIC = 'MAP'
MAP_PORT = '5559'

STATE_TOPIC = 'STATE'
STATE_PORT = '5558'

ACTION_TOPIC = 'ACTION'
ACTION_PORT = '5555'

CONTROL_TOPIC = 'CONTROL'
CONTROL_PORT = '5554'

ENABLE_ET_GOA = True
ENABLE_GOA = True

LOGGING_PATH = './data/logs'
SCENARIO_PATH = './data/scenarios'

CARGO_RESUPPLY = 3

SCENARIO_ID = 7

#
# Agent IDs and colors
#

AGENT1_ID = 1
AGENT1_COLOR = (0, 102, 255)
AGENT2_ID = 2
AGENT2_COLOR = (255, 204, 0)  # light blue
AGENT3_ID = 3
AGENT3_COLOR = (0, 255, 0)

AGENT_COLOR = {AGENT1_ID: AGENT1_COLOR,
               AGENT2_ID: AGENT2_COLOR,
               AGENT3_ID: AGENT3_COLOR}

#
# Initial environment and state
#
OBSTRUCTION_COLOR = (0, 0, 0)
ZONE_COLOR = (0, 0, 255)
CRATER_COLOR = (255, 255, 255)

#
# Control policies
#
OA_ROLLOUTS = 25
STATE_UNCERTAINTY = 0.9
ZONE_UNCERTAINTY = 0.5
ET_THRESHOLD = 0.05

#
# Control delays
#
SPEED_AUTONOMY = 0.5


class Location:
    def __init__(self, name, position, policy):
        self.name = name
        self.position = position
        self.policy = policy


HOME = 'HOME'
AREA_1 = 'AREA_1'
AREA_2 = 'AREA_2'
AREA_3 = 'AREA_3'
LOCATIONS = {HOME: Location(HOME, np.asarray([10, 10]), './solvers/q_HOME.npy'),
             AREA_1: Location(AREA_1, np.asarray([45, 20]), './solvers/q_AREA_1.npy'),
             AREA_2: Location(AREA_2, np.asarray([14, 40]), './solvers/q_AREA_2.npy'),
             AREA_3: Location(AREA_3, np.asarray([38, 47]), './solvers/q_AREA_3.npy')}

#
# FaMSeC
#
FAMSEC_COLORS = {'Highly Unlikely': 'darkRed',
                 'Unlikely': 'darksalmon',
                 'About Even': 'yellow',
                 'Likely': 'greenyellow',
                 'Highly Likely': 'darkGreen'}
FAMSEC_LABELS = {'Highly Unlikely': [0.0, 0.1],
                 'Unlikely': [0.1, 0.4],
                 'About Even': [0.4, 0.6],
                 'Likely': [0.6, 0.9],
                 'Highly Likely': [0.90, 1.1]}


def convert_famsec(prob):
    """
    https://waf.cs.illinois.edu/visualizations/Perception-of-Probability-Words/

    :param prob:
    :return:
    """

    for (k, v) in FAMSEC_LABELS.items():
        if v[0] <= prob < v[1]:
            return k


class MultiAgentState:
    # constant dict keys
    STATUS_AGENTID = 'STATUS_AGENTID'
    STATUS_LOCATION = 'STATUS_LOCATION'
    STATUS_STATE = 'STATUS_STATE'
    STATUS_SIM_TIME = 'STATUS_SIM_TIME'
    STATUS_HITS = 'STATUS_HITS'
    STATUS_GOAL = 'STATUS_GOAL'
    STATUS_DELIVERIES = 'STATUS_DELIVERIES'
    STATUS_CARGO_COUNT = 'STATUS_CARGO_COUNT'
    STATUS_ASSESSED_GOAL = 'STATUS_ASSESSED_GOAL'
    STATUS_REWARDS = 'STATUS_REWARDS'
    STATUS_ZONES = 'STATUS_ZONES'
    STATUS_PREDICTED_ZONES = 'STATUS_PREDICTED_ZONES'
    STATUS_CRATERS = 'STATUS_CRATERS'
    STATUS_PREDICTED_CRATERS = 'STATUS_PREDICTED_CRATERS'
    STATUS_COLOR = 'STATUS_COLOR'
    STATUS_REWARD_GOA = 'STATUS_REWARD_GOA'
    STATUS_COLLISIONS_GOA = 'STATUS_COLLISIONS_GOA'
    STATUS_ZONES_GOA = 'STATUS_TIME_GOA'

    # constant state enums
    STOP = 'STOP'
    START = 'START'
    ASSESSMENT = 'ASSESSMENT'
    FULL_ASSESSMENT = 'FULL_ASSESSMENT'
    END_SIM = 'END_SIM'

    def __init__(self, agent_id, start_time, agent_color):
        self.agent_id = agent_id  # AGENTxx
        self.movement_state = MultiAgentState.STOP  # START/STOP
        self.location = [0, 0]  # (X,Y)
        self.goal = HOME  # AREA1, AREA2, AREA3, HOME
        self.color = agent_color  # RED, BLUE, GREEN
        self.sim_start_time = start_time  # Integer
        self.sim_current_time = 0  # Integer
        self.needs_assessment = False
        self.needs_end_sim = False
        self.at_goal = True
        self.delivery_count = 0
        self.cargo_count = CARGO_RESUPPLY

        self.current_event_id = 0

        self.assessed_goal = HOME

        self.reward_assessment_threshold = 0
        self.reward_assessment = 1  # String
        self.reward_count = 0

        self.collision_assessment_threshold = 0  # String
        self.collision_assessment = 1  # String
        self.collision_count = 0  # Integer
        self.predicted_collision_count = (0, 0)

        self.zone_assessment_threshold = 0  # String
        self.zone_assessment = 1  # String
        self.zone_count = 0
        self.predicted_zone_count = (0, 0)

    def state_update_message(self):
        single_agent_msg = {self.STATUS_AGENTID: self.agent_id,
                            self.STATUS_STATE: self.movement_state,
                            self.STATUS_LOCATION: tuple([int(x) for x in self.location]),
                            self.STATUS_HITS: self.collision_count,
                            self.STATUS_GOAL: self.goal,
                            self.STATUS_SIM_TIME: self.sim_current_time,
                            self.STATUS_REWARDS: self.reward_count,
                            self.STATUS_ASSESSED_GOAL: self.assessed_goal,
                            self.STATUS_ZONES: self.zone_count,
                            self.STATUS_PREDICTED_ZONES: self.predicted_zone_count,
                            self.STATUS_CRATERS: self.collision_count,
                            self.STATUS_PREDICTED_CRATERS: self.predicted_collision_count,
                            self.STATUS_COLOR: self.color,
                            self.STATUS_REWARD_GOA: convert_famsec(self.reward_assessment),
                            self.STATUS_COLLISIONS_GOA: convert_famsec(self.collision_assessment),
                            self.STATUS_ZONES_GOA: convert_famsec(self.zone_assessment),
                            self.STATUS_DELIVERIES: self.delivery_count,
                            self.STATUS_CARGO_COUNT: self.cargo_count}
        return single_agent_msg


class MessageHelpers:
    TOPICS_END_SIM = 'END_SIM'
    TOPICS_ASSESSMENT_REQUEST = 'ASSESSMENT_REQUEST'
    TOPICS_MOVE_REQUEST = 'MOVE_REQUEST'
    TOPICS_GOAL_REQUEST = 'GOAL_REQUEST'
    TOPICS_STATE_UPDATE = 'STATE_UPDATE'
    TOPICS_EVENT_UPDATE = 'EVENT_UPDATE'

    @staticmethod
    def unpack(msg):
        msg = json.loads(msg)
        return msg['topic'], msg['data']

    @staticmethod
    def pack_message(topic, msg):
        msg = {'topic': topic, 'data': msg}
        msg_json = json.dumps(msg)
        return msg_json

    @staticmethod
    def end_sim():
        return MessageHelpers.pack_message(MessageHelpers.TOPICS_END_SIM, '')

    @staticmethod
    def state_update(msg):
        """ to UI from backend """
        return MessageHelpers.pack_message(MessageHelpers.TOPICS_STATE_UPDATE, msg)

    @staticmethod
    def assessment_request(agent_id, reward_threshold, zone_threshold, collision_threshold):
        """ to backend from UI """
        msg = {'AGENTID': agent_id,
               'REWARD': reward_threshold,
               'ZONE': zone_threshold,
               'COLLISIONS': collision_threshold}
        return MessageHelpers.pack_message(MessageHelpers.TOPICS_ASSESSMENT_REQUEST, msg)

    @staticmethod
    def unpack_assessment_request(msg):
        """ at backend """
        return msg['AGENTID'], msg['REWARD'], msg['ZONE'], msg['COLLISIONS']

    @staticmethod
    def move_request(agent_id, state):
        """ to backend from UI """
        msg = {'AGENTID': agent_id,
               'STATE': state}
        return MessageHelpers.pack_message(MessageHelpers.TOPICS_MOVE_REQUEST, msg)

    @staticmethod
    def unpack_move_request(msg):
        """ at backend """
        return msg['AGENTID'], msg['STATE']

    @staticmethod
    def goal_request(agent_id, new_goal):
        """ to backend from UI """
        msg = {'AGENTID': agent_id,
               'GOAL': new_goal}
        return MessageHelpers.pack_message(MessageHelpers.TOPICS_GOAL_REQUEST, msg)

    @staticmethod
    def unpack_goal_request(msg):
        """ at backend """
        return msg['AGENTID'], msg['GOAL']

    @staticmethod
    def map_out(image_array):
        return base64.b64encode(cv2.imencode('.png', image_array)[1]).decode()

    @staticmethod
    def map_in(image_string):
        jpg_original = base64.b64decode(image_string)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        image_array = cv2.imdecode(jpg_as_np, flags=1)
        return image_array

    @staticmethod
    def for_me(msg, agent_id):
        return msg['AGENTID'] == agent_id

    @staticmethod
    def pack_event(event_id):
        msg = {'EVENT_ID': event_id}
        return MessageHelpers.pack_message(MessageHelpers.TOPICS_EVENT_UPDATE, msg)

    @staticmethod
    def unpack_event(msg):
        """ at all nodes"""
        return msg['EVENT_ID']


class Obstacle:
    def __init__(self, x, y, radius, color):
        self.xy = np.asarray([x, y])
        self.r = radius
        self.color = color

    def collision(self, x, y):
        d = np.linalg.norm(self.xy - np.asarray([x, y]))
        if d < self.r:
            return True
        else:
            return False


class Event:
    def __init__(self, event_time, changed_zones=None, changed_craters=None):
        self.time = event_time
        self.changed_zones = [] if changed_zones is None else changed_zones
        self.changed_craters = [] if changed_craters is None else changed_craters


def create_new_craters():
    lx = np.random.randint(10, 50)
    ly = np.random.randint(10, 50)
    sx = np.random.randint(2, 4)
    sy = np.random.randint(2, 4)
    craters = [Obstacle(int(x), int(y), 1, CRATER_COLOR) for (x, y) in
               zip(np.random.normal(loc=lx, scale=sx, size=25), np.random.normal(loc=ly, scale=sy, size=25))]

    sx = sx+3
    sy = sx+3
    zones = [Obstacle(int(x), int(y), 2, ZONE_COLOR) for (x, y) in
               zip(np.random.normal(loc=lx, scale=sx, size=25), np.random.normal(loc=ly, scale=sy, size=25))]
    return craters, zones


def save_scenarios(num_runs):
    for i in range(num_runs):
        craters, zones = create_new_craters()
        craters_out = np.zeros((len(craters), 3))
        zones_out = np.zeros((len(zones), 3))
        for c in range(len(craters)):
            craters_out[c] = [*craters[c].xy, craters[c].r]
        for z in range(len(zones)):
            zones_out[z] = [*zones[z].xy, zones[z].r]
        np.save('./data/scenarios/craters_scenario_{}.npy'.format(i), craters_out)
        np.save('./data/scenarios/zones_scenario_{}.npy'.format(i), zones_out)


def read_scenarios(scenario_id):
    c1, z1 = read_scenario(scenario_id)
    c2, z2 = read_scenario(scenario_id+1)
    return c1+c2, z1+z2


def read_scenario(scenario_id):
    c = np.load('./data/scenarios/craters_scenario_{}.npy'.format(scenario_id))
    z = np.load('./data/scenarios/zones_scenario_{}.npy'.format(scenario_id))
    craters = [Obstacle(int(x), int(y), int(r), CRATER_COLOR) for (x, y, r) in c]
    zones = [Obstacle(int(x), int(y), int(r), ZONE_COLOR) for (x, y, r) in z]
    return craters, zones
