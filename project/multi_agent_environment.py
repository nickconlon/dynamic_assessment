import numpy as np
import environment as single_agent_environment
import cv2
import copy
import time

import project.solvers.q_learning_policy as policy
import project.multiagent_configs as configs

class Agent:
    def __init__(self, agent_id, agent_color, current_location, goal_label, policy_paths, policy_labels, _obstacles=None, _zones=None,
                 _craters=None):
        self.agent_id = agent_id
        self.start = time.time()
        self.color = agent_color
        self.current_location = copy.deepcopy(current_location)
        self.goal_label = goal_label
        self.policy = policy.q_policies(policy_paths, policy_labels)
        self.env = single_agent_environment.Environment(self.current_location, configs.LOCATIONS[goal_label].position,
                                                        _obstacles=_obstacles, _zones=_zones, _craters=_craters)
        self.done = False

    def event(self, craters_changed=None, zones_changed=None, new_goal_label=None):
        self.env.change_event(new_craters=craters_changed,
                              new_zones=zones_changed,
                              new_goal_label=new_goal_label)

        if new_goal_label is not None:
            self.goal_label = new_goal_label

    def step(self):
        _action = self.policy.get_policy(self.env.index_from_xy(*self.current_location), self.goal_label)
        _state_index, _reward, _done, _info = self.env.step(_action)
        self.current_location = self.env.xy_from_index(_state_index)
        self.done |= _done

    def is_done(self):
        return self.done

    def get_state(self):
        s = configs.MultiAgentState(self.agent_id, self.start, self.color)
        s.location = self.current_location
        s.goal = self.goal_label
        return s.state_update_message()


class MultiAgentRendering(single_agent_environment.Environment):
    def __init__(self, _pos_start, _goal):
        super().__init__(_pos_start, _goal)
        self.previous_positions = {1:[], 2:[], 3:[]}  # {agentID: [[],...,[]]
        self.multiagent_states = []

    @staticmethod
    def draw_shapes(img, position, color):
        """
        Shaded areas for obstacles.
        """
        shapes = img.copy()
        scale = 10
        FOV = 3
        shapes = cv2.circle(shapes, (position[0] * scale, position[1] * scale), FOV * scale, color, cv2.FILLED)
        out = img.copy()
        alpha = 0.4
        cv2.addWeighted(shapes, alpha, img, 1 - alpha, 0, out)

        return out

    def render(self, mode="human"):
        """
        Render the environment.
        """
        _image = np.copy(self.base_image)
        scale = 10
        for agent_state in self.multiagent_states:
            agent_id = agent_state[configs.MultiAgentState.STATUS_AGENTID]
            agent_location = agent_state[configs.MultiAgentState.STATUS_LOCATION]
            agent_color = agent_state[configs.MultiAgentState.STATUS_COLOR]
            agent_goal = agent_state[configs.MultiAgentState.STATUS_GOAL]
            agent_goal = configs.LOCATIONS[agent_goal].position
            self.previous_positions[agent_id].append(agent_location)

            for p in self.previous_positions[agent_id]:
                cv2.circle(_image, (p[0] * scale, p[1] * scale), 6, agent_color, thickness=-1)

            _image = cv2.circle(_image, (agent_location[0] * scale, agent_location[1] * scale), 5, (0, 0, 0), thickness=-1)
            _image = self.draw_shapes(_image, agent_location, agent_color)

            _image = cv2.circle(_image, (agent_goal[0] * scale, agent_goal[1] * scale), self.goal_eps * scale,
                                self.goal_color, thickness=2)

        _image = cv2.circle(_image, (self.pos_home[0] * scale, self.pos_home[1] * scale), self.goal_eps * scale,
                            self.home_color, thickness=2)

        for obstacle in self.obstacles:
            _image = cv2.circle(_image, (obstacle.xy[0] * scale, obstacle.xy[1] * scale), obstacle.r * scale,
                                obstacle.color, thickness=2)

        for zone in self.zones:
            _image = cv2.circle(_image, (zone.xy[0] * scale, zone.xy[1] * scale), zone.r * scale, zone.color,
                                thickness=2)

        for crater in self.craters:
            _image = cv2.circle(_image, (crater.xy[0] * scale, crater.xy[1] * scale), crater.r * scale, crater.color,
                                thickness=2)

        _image = cv2.rectangle(_image, (self.minX * scale, self.minY * scale),
                               (self.maxX * scale, self.maxY * scale), (0, 0, 0), thickness=2)
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


if __name__ == '__main__':
    from project.solvers.q_learning_policy import q_policies

    initial_state = configs.LOCATIONS[configs.HOME].position
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

    a1 = Agent(1, (0, 0, 255), initial_state, configs.AREA_1, available_policies, available_target_names, _obstacles=obstacles, _zones=zones, _craters=craters)
    a2 = Agent(2, (255, 0, 0), initial_state, configs.AREA_3, available_policies, available_target_names, _obstacles=obstacles, _zones=zones, _craters=craters)
    a3 = Agent(3, (0, 255, 0), initial_state, configs.AREA_2, available_policies, available_target_names, _obstacles=obstacles, _zones=zones, _craters=craters)
    rendering = MultiAgentRendering(initial_state, [])

    for i in range(100):
        if i == 20:
            a1.event(new_goal_label=configs.AREA_3)

        should_stop = True
        states = []
        for a in [a1, a2, a3]:
            if a.is_done():
                continue
            else:
                a.step()
                should_stop = False
        states.append(a1.get_state())

        rendering.multiagent_states = states
        rendering.render(mode='human')

        if should_stop:
            break