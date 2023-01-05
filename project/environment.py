import numpy as np
import copy
from gym import Env, spaces
import cv2
import project.multiagent_configs as configs
from project.multiagent_configs import Obstacle
import logging


class Environment(Env):
    def __init__(self, _pos_start, _goal_label, img_path=None,
                 _obstacles=None, _zones=None, _craters=None,
                 _state_transition=1.0, _zone_transition=0.5,
                 _prob_avoiding_crater=0.0, _prob_avoiding_zone=0.0):
        super(Environment, self).__init__()
        self.minX = 0
        self.maxX = 50
        self.minY = 0
        self.maxY = 50
        self.step_size = 10
        self.scale = 10
        self.agent_FOV = 3

        """ Gym stuff """
        self.observation_space = spaces.Discrete(self.maxX * self.maxY)
        self.action_space = spaces.Discrete(4)

        """ Our stuff """
        self.actions = {0: np.asarray([-1, 0]),
                        1: np.asarray([+1, 0]),
                        2: np.asarray([0, +1]),
                        3: np.asarray([0, -1])}
        self.goal_eps = 2
        self.goal = copy.deepcopy(configs.LOCATIONS[_goal_label].position)
        self.goal_label = _goal_label
        self.pos_start = copy.deepcopy(_pos_start)
        self.pos_home = copy.deepcopy(_pos_start)
        self.pos = copy.deepcopy(self.pos_start)

        self.previous_pos = []
        self.obstacles = _obstacles if _obstacles is not None else []
        self.zones = _zones if _zones is not None else []
        self.craters = _craters if _craters is not None else []

        """ Rendering in BGR"""
        # TODO this should be an argument or figure out how to do relative pathing
        if img_path is None:
            self.base_image = cv2.imread("./imgs/mars.jpg")
        else:
            self.base_image = cv2.imread(img_path)
        self.path_color = (107, 183, 189)
        self.goal_color = (0, 255, 0)
        self.home_color = (0, 255, 0)

        self.stochastic_transitions = False
        self.state_transition_probability = _state_transition
        self.zone_transition_probability = _zone_transition
        self.probability_avoiding_crater = _prob_avoiding_crater
        self.probability_avoiding_zone = _prob_avoiding_zone

        self.num_steps = 0

    def xy_from_index(self, index):
        """
        Convert from an index to (x,y).
        """
        x = index % self.maxX
        y = index / self.maxX
        return int(x), int(y)

    def index_from_xy(self, x, y):
        """
        Convert from an (x,y) to an index.
        """
        index = x + self.maxX * y
        return int(index)

    def obstacles_in_fov(self, obstacles):
        zone_count = 0
        for obstacle in obstacles:
            d = np.linalg.norm(obstacle.xy - np.asarray(self.pos))
            if d <= (obstacle.r + self.agent_FOV):
                zone_count += 1
        return zone_count

    def step(self, action):
        """
        Execute the action from the current position in this environment.
        """
        craters_hit = 0
        zone_hits = 0
        # Base transition
        if self.stochastic_transitions:
            if np.random.rand() > self.state_transition_probability:
                possible_actions = list(self.actions.keys())
                possible_actions.remove(action)
                action = np.random.choice(a=possible_actions, p=np.ones_like(possible_actions) / len(possible_actions))

        # Slippery zones impact transition
        for zone in self.zones:
            if zone.collision(*self.pos):
                if np.random.rand() > self.probability_avoiding_zone:
                    zone_hits = 1
                    if np.random.rand() > self.zone_transition_probability:
                        possible_actions = list(self.actions.keys())
                        possible_actions.remove(action)
                        action = np.random.choice(a=possible_actions,
                                                  p=np.ones_like(possible_actions) / len(possible_actions))

        # Make the transition
        if self.valid_action(action):
            self.previous_pos.append(np.copy(self.pos))
            self.pos += self.actions[action]

        # Count up the craters we hit with the transition

        for crater in self.craters:
            if crater.collision(*self.pos):
                if np.random.rand() > self.probability_avoiding_crater:
                    craters_hit = 1

        self.num_steps += 1

        _reward = -1 if not self.captured_goal() else +10
        _done = False if not self.captured_goal() else True
        _info = {'collisions': craters_hit,
                 'zones': zone_hits,
                 'times': self.num_steps,
                 'rewards': _reward,
                 'location': self.goal_label,
                 'zones_seen': self.obstacles_in_fov(self.zones),
                 'craters_seen': self.obstacles_in_fov(self.craters)}
        return self.index_from_xy(*self.pos), _reward, _done, _info

    def captured_goal(self):
        """
        Return true if the goal has been captured, false otherwise
        """
        return np.linalg.norm(self.pos - self.goal) <= self.goal_eps

    def valid_action(self, action):
        """
        Return true if the action is valid, false otherwise
        """
        _sp = self.pos + self.actions[action]
        _valid = False
        """ Is the agent within the map bounds?"""
        if self.minX <= _sp[0] < self.maxX and self.minY <= _sp[1] < self.maxY:
            _valid = True

        """" is the agent on an obstacle?"""
        for obstacle in self.obstacles:
            if obstacle.collision(_sp[0], _sp[1]):
                _valid = False
        return _valid

    def change_event(self, new_craters=None, new_zones=None, new_goal_label=None):
        """
        Change some element(s) of the environment.
        """

        if new_craters is not None:
            self.craters = copy.deepcopy(new_craters)
        if new_zones is not None:
            self.zones = copy.deepcopy(new_zones)
        if new_goal_label is not None:
            self.goal = copy.deepcopy(configs.LOCATIONS[new_goal_label].position)
            self.goal_label = new_goal_label

    def apply_event(self, event, time):
        """
        Apply an event to this environment. Return 1 if the event was applied, 0 otherwise.
        """
        if event.time == time:
            self.change_event(new_craters=event.changed_craters, new_zones=event.changed_zones)
            return 1
        else:
            return 0

    def reset(self, state=None):
        """
        Reset the environment, optionally using the given state.
        """
        if state is None:
            self.pos = copy.deepcopy(self.pos_start)
        else:
            self.pos = copy.deepcopy(state)
        self.previous_pos = []
        self.num_steps = 0
        return self.index_from_xy(*self.pos)

    #
    # Rendering stuff
    #
    @staticmethod
    def draw_grid(_image, scale):
        """
        Draw grid lines on the rendered image.
        """
        # vertical lines
        minn = 0
        maxx = 50 * scale

        for x in np.arange(minn, maxx, scale):
            _image = cv2.line(_image, (x + 5, minn + 5), (x + 5, maxx + 5), color=(0, 0, 0), thickness=2)
        # horizontal lines
        for y in np.arange(minn, maxx, scale):
            _image = cv2.line(_image, (minn + 5, y + 5), (maxx + 5, y + 5), color=(0, 0, 0), thickness=2)
        return _image

    def draw_shapes(self, img, position, color):
        """
        Shaded areas for obstacles.
        """
        shapes = img.copy()
        shapes = cv2.circle(shapes, (position[0] * self.scale, position[1] * self.scale),
                            self.agent_FOV * self.scale, color, cv2.FILLED)
        out = img.copy()
        alpha = 0.4
        cv2.addWeighted(shapes, alpha, img, 1 - alpha, 0, out)

        return out

    def render(self, mode="human"):
        """
        Render the environment.
        """
        _image = np.copy(self.base_image)
        for p in self.previous_pos:
            cv2.circle(_image, (p[0] * self.scale, p[1] * self.scale), 6, self.path_color, thickness=-1)

        _image = cv2.circle(_image, (self.pos[0] * self.scale, self.pos[1] * self.scale), 5, (0, 0, 0), thickness=-1)

        _image = cv2.circle(_image, (self.goal[0] * self.scale, self.goal[1] * self.scale), self.goal_eps * self.scale,
                            self.goal_color, thickness=2)

        _image = cv2.circle(_image, (self.pos_home[0] * self.scale, self.pos_home[1] * self.scale), self.goal_eps * self.scale,
                            self.home_color, thickness=2)

        _image = self.draw_shapes(_image, self.pos, (0,0,255))

        for obstacle in self.obstacles:
            _image = cv2.circle(_image, (obstacle.xy[0] * self.scale, obstacle.xy[1] * self.scale), obstacle.r * self.scale,
                                obstacle.color, thickness=2)

        for zone in self.zones:
            _image = cv2.circle(_image, (zone.xy[0] * self.scale, zone.xy[1] * self.scale), zone.r * self.scale, zone.color,
                                thickness=2)

        for crater in self.craters:
            _image = cv2.circle(_image, (crater.xy[0] * self.scale, crater.xy[1] * self.scale), crater.r * self.scale, crater.color,
                                thickness=2)

        _image = cv2.rectangle(_image, (self.minX * self.scale, self.minY * self.scale),
                               (self.maxX * self.scale, self.maxY * self.scale), (0, 0, 0), thickness=2)
        _image = _image[0:600, 0:600, :]
        # _image = self.draw_grid(_image)
        # _image = self.draw_shapes(_image)
        if mode == "human":
            _image = cv2.resize(_image, (500, 500), interpolation=cv2.INTER_AREA)
            cv2.imshow('test', _image)
            cv2.waitKey(100)
        elif mode == "rgb_array":
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
            return _image

    def greedy_policy(self, _eps):
        """
        Epsilon greedy policy.
        """
        _p = np.random.rand()
        if _p > _eps:
            action = np.random.randint(0, len(self.actions))
        else:
            _diff = self.pos - self.goal
            if abs(_diff[0]) > abs(_diff[1]):
                if _diff[0] > 0:
                    action = 0
                else:
                    action = 1
            else:
                if _diff[1] > 0:
                    action = 3
                else:
                    action = 2
        return action


if __name__ == '__main__':
    from project.solvers.q_learning_policy import q_policies

    initial_state = configs.LOCATIONS[configs.HOME].position
    goal_location = configs.LOCATIONS[configs.AREA_1].position
    obstacles = []
    zones = []
    craters = []

    available_policies = [configs.LOCATIONS[configs.HOME].policy,
                          configs.LOCATIONS[configs.AREA_1].policy,
                          configs.LOCATIONS[configs.AREA_2].policy,
                          configs.LOCATIONS[configs.AREA_3].policy]
    available_target_names = [configs.LOCATIONS[configs.HOME].name,
                              configs.LOCATIONS[configs.AREA_1].name,
                              configs.LOCATIONS[configs.AREA_2].name,
                              configs.LOCATIONS[configs.AREA_3].name]
    policy_container = q_policies(available_policies, available_target_names)

    env = Environment(initial_state, configs.AREA_1, _obstacles=obstacles, _zones=zones)

    zones1 = [Obstacle(int(x), int(y), 2, configs.ZONE_COLOR) for (x, y) in
              zip(np.random.normal(loc=40, scale=5, size=25), np.random.normal(loc=20, scale=5, size=25))]
    zones2 = [Obstacle(int(x), int(y), 2, configs.ZONE_COLOR) for (x, y) in
              zip(np.random.normal(loc=40, scale=5, size=25), np.random.normal(loc=20, scale=5, size=25))]
    s = env.reset()
    logging.basicConfig(filename='test.log',format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    for i in range(100):
        logging.warning("test")
        env.render()
        a = policy_container.get_policy(s, configs.AREA_1)
        s, r, done, info = env.step(a)
        print(r)
        if done:
            env.render()
            break
