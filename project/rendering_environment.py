import numpy as np
import environment as single_agent_environment
import traceback
import cv2
import copy
import time

import project.solvers.q_learning_policy as policy
import project.multiagent_configs as configs
from project.assessment.assessments import StaticAssessment, DynamicAssessment


class Agent:
    def __init__(self, agent_id, agent_color, current_location, goal_label, policy_paths, policy_labels,
                 _obstacles=None, _zones=None,
                 _craters=None):
        self.agent_id = agent_id
        self.start = time.time()
        self.color = agent_color
        self.current_location = copy.deepcopy(current_location)
        self.goal_label = goal_label
        self.policy = policy.q_policies(policy_paths, policy_labels)
        self.env = single_agent_environment.Environment(self.current_location, self.goal_label,
                                                        _obstacles=_obstacles, _zones=_zones, _craters=_craters)
        self.done = False
        self.static_assessment = StaticAssessment()
        self.dynamic_assessment = DynamicAssessment()
        self.zones = 0
        self.craters = 0
        self.assessments = 0

        self.predicted_state = np.array([[self.current_location]])
        self.assessment_index = 1
        self.et_threshold = 0.05

        self.predicted_zones_seen = np.array([[0]])
        self.predicted_craters_seen = np.array([[0]])
        self.actual_craters_seen = {0: 0}
        self.actual_zones_seen = {0: 0}

        self.timestep = 0

    def dynamic_assess(self, goals):
        should_assess = False
        if self.assessment_index >= self.predicted_state.shape[1] or np.count_nonzero(
                np.isnan(self.predicted_state[:, self.assessment_index])):
            should_assess = True
            print('assessing due to lack of data')
        else:
            predicted_craters_t = self.predicted_craters_seen[:, self.assessment_index+1]
            observed_craters_t = self.actual_craters_seen[self.timestep]
            p_craters = self.dynamic_assessment.normal_surprise_1d(predicted_craters_t, observed_craters_t)

            predicted_zones_t = self.predicted_zones_seen[:, self.assessment_index + 1]
            observed_zones_t = self.actual_zones_seen[self.timestep]
            p_zones = self.dynamic_assessment.normal_surprise_1d(predicted_zones_t, observed_zones_t)
            p_expected = p_craters & p_zones
            '''
            try:
                predicted_state_t = copy.deepcopy(self.predicted_state[:, self.assessment_index])
                predicted_state_t = predicted_state_t[~np.isnan(predicted_state_t).any(axis=1), :]
                p_expected = self.dynamic_assessment.tail(predicted_state_t, self.current_location)
                print("tail: {:.2f}".format(p_expected))
            except Exception as e:
                p_expected = 0.0
                print(e)
                traceback.print_exc()
                '''
            if p_expected <= self.et_threshold:
                should_assess = True
                print('assessing due to surprising data')

        if should_assess:
            self.assessment_index = 1
            return self.choose_goal(goals)
        self.assessment_index += 1
        return None

    def choose_goal(self, goals, printing=True):
        self.assessments += 1
        scores = np.zeros(len(goals))
        reports = [None, None, None]
        for goal_idx in range(len(goals)):
            goa_report = self._assess(goals[goal_idx])
            reports[goal_idx] = goa_report
            scores[goal_idx] = goa_report.delivery_goa

        best_zone_idx = np.argmax(scores)
        report = reports[best_zone_idx]
        self.predicted_state = report.predicted_states
        self.predicted_zones_seen = report.predicted_dust_fov
        self.predicted_craters_seen = report.predicted_craters_fov
        mu_craters = report.mu_craters
        std_craters = report.std_craters
        mu_zones = report.mu_zones
        std_zones = report.std_zones

        if printing:
            print('The likelihood of successful delivery to area {}: {}'.format(best_zone_idx,
                                                                                configs.AssessmentReport.convert_famsec(
                                                                                    scores[best_zone_idx])))
            print('Because the rover will hit:')
            print(' {}'.format(int(mu_craters)) + u" \u00B1 " + '{} craters'.format(int(std_craters)))
            print(' {}'.format(int(mu_zones)) + u" \u00B1 " + '{} zones'.format(int(std_zones)))
            print('\n')
            print(scores)
        return best_zone_idx, [configs.AssessmentReport.convert_famsec(s) for s in scores], scores

    def event(self, new_craters=None, new_zones=None, new_goal_label=None):
        self.env.change_event(new_craters=new_craters, new_zones=new_zones, new_goal_label=new_goal_label)
        if new_goal_label is not None:
            self.goal_label = new_goal_label

    def step(self):
        _action = self.policy.get_policy(self.env.index_from_xy(*self.current_location), self.goal_label)
        _state_index, _reward, _done, _info = self.env.step(_action)
        self.current_location = self.env.xy_from_index(_state_index)
        self.done |= _done
        self.zones += _info['zones']
        self.craters += _info['collisions']
        self.actual_craters_seen[_info['times']] = _info['craters_seen']
        self.actual_zones_seen[_info['times']] = _info['zones_seen']
        self.timestep = _info['times']

    def reset(self, location):
        self.current_location = copy.deepcopy(location)
        self.done = False
        self.env.reset(state=location)
        self.zones = 0
        self.craters = 0
        self.assessment_index = 1
        self.assessments = 0

    def is_done(self):
        return self.done

    def get_state(self):
        s = configs.MultiAgentState(self.agent_id, self.start, self.color)
        s.location = self.current_location
        s.goal = self.goal_label
        return s.state_update_message()

    def _assess(self, goal_label):
        copy_env = copy.deepcopy(self.env)
        copy_env.goal = copy.deepcopy(configs.LOCATIONS[goal_label].position)
        report = self.static_assessment.run_another_assessment(
            self.policy.policies[goal_label],
            copy_env,
            self.current_location,
            configs.OA_ROLLOUTS,
            configs.STATE_UNCERTAINTY,
            craters_already_hit=self.craters,
            zones_already_hit=self.zones)

        return report

    def rollout(self, goal_label):
        report = self.static_assessment.rollout_all(
            self.policy.policies[goal_label],
            copy.deepcopy(self.env),
            self.current_location,
            25, 0.9)
        return report


class MultiAgentRendering(single_agent_environment.Environment):
    def __init__(self, agent_ids):
        super().__init__(configs.LOCATIONS[configs.HOME].position, configs.HOME)
        self.agent_ids = copy.deepcopy(agent_ids)
        self.previous_positions = {}
        self.states = {}
        self.reset()

    def state_update(self, state_msg):
        agent_id = state_msg[configs.MultiAgentState.STATUS_AGENTID]
        if agent_id in self.states:
            self.states[agent_id] = state_msg

    @staticmethod
    def add_convex_hull(_image, _obstacles, _color):
        imgg = np.zeros_like(_image)
        _img = _image.copy()
        for obs in _obstacles:
            imgg = cv2.circle(imgg, (obs.xy[0] * 10, obs.xy[1] * 10), obs.r * 10, obs.color, thickness=2)

        gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
        kernel_dilation = np.ones((4, 4), np.uint8)
        gray = cv2.dilate(gray, kernel_dilation, iterations=12)
        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # create hull array for convex hull points
        hull = []
        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))

        for i in range(len(contours)):
            _img = cv2.drawContours(_img, hull, i, _color, -1)

        alpha = 0.4
        _img = cv2.addWeighted(_img, alpha, _image, 1 - alpha, 0)

        return hull, contours, _img

    def render(self, mode="human"):
        """
        Render the environment.
        """
        _image = np.copy(self.base_image)
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

        _, _, _image = self.add_convex_hull(_image, self.zones, (22, 22, 138))
        _, _, _image = self.add_convex_hull(_image, self.craters, (255, 0, 0))

        for obstacle in self.obstacles:
            _image = cv2.circle(_image, (obstacle.xy[0] * self.scale, obstacle.xy[1] * self.scale), obstacle.r * self.scale,
                                obstacle.color, thickness=2)

        _image = cv2.rectangle(_image, (self.minX * self.scale, self.minY * self.scale),
                               (self.maxX * self.scale, self.maxY * self.scale), (0, 0, 0), thickness=2)

        _image = cv2.circle(_image, (self.pos_home[0] * self.scale, self.pos_home[1] * self.scale), self.goal_eps * self.scale,
                            self.home_color, thickness=1)
        _image = cv2.putText(_image, 'H', (self.pos_home[0] * self.scale - 10, self.pos_home[1] * self.scale + 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        for g in [configs.AREA_1, configs.AREA_2, configs.AREA_3]:
            pos = configs.LOCATIONS[g].position
            _image = cv2.circle(_image, (pos[0] * self.scale, pos[1] * self.scale), self.goal_eps * self.scale,
                                self.goal_color, thickness=1)
            _image = cv2.putText(_image, g.split('_')[1], (pos[0] * self.scale - 10, pos[1] * self.scale + 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        for agent_id in self.agent_ids:
            if agent_id in self.states:
                agent_state = self.states[agent_id]
                if agent_state:
                    agent_location = agent_state[configs.MultiAgentState.STATUS_LOCATION]
                    agent_color = agent_state[configs.MultiAgentState.STATUS_COLOR]
                    agent_goal = agent_state[configs.MultiAgentState.STATUS_GOAL]
                    agent_goal = configs.LOCATIONS[agent_goal].position
                    self.previous_positions[agent_id].append(agent_location)

                    for p in self.previous_positions[agent_id]:
                        cv2.circle(_image, (p[0] * self.scale, p[1] * self.scale), 6, agent_color, thickness=-1)

                    _image = cv2.circle(_image, (agent_location[0] * self.scale, agent_location[1] * self.scale), 5, (0, 0, 0),
                                        thickness=-1)
                    _image = self.draw_shapes(_image, agent_location, agent_color)

                    _image = cv2.circle(_image, (agent_goal[0] * self.scale, agent_goal[1] * self.scale), self.goal_eps * self.scale,
                                        self.goal_color, thickness=3)

        _image = _image[0:600, 0:600, :]

        if mode == "human":
            _image = cv2.resize(_image, (500, 500), interpolation=cv2.INTER_AREA)
            cv2.imshow('test', cv2.cvtColor(_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(100)
        elif mode == "rgb_array":
            #_image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
            return _image.copy()

    def reset(self):
        super().reset()
        for agent_id in self.agent_ids:
            self.previous_positions[agent_id] = []
            self.states[agent_id] = {}


if __name__ == '__main__':
    from project.solvers.q_learning_policy import q_policies

    initial_state = configs.LOCATIONS[configs.HOME].position
    obstacles = []
    craters, zones = configs.read_scenarios(configs.SCENARIO_ID)

    available_policies = [configs.LOCATIONS[configs.HOME].policy,
                          configs.LOCATIONS[configs.AREA_1].policy,
                          configs.LOCATIONS[configs.AREA_2].policy,
                          configs.LOCATIONS[configs.AREA_3].policy]
    available_target_names = [configs.LOCATIONS[configs.HOME].name,
                              configs.LOCATIONS[configs.AREA_1].name,
                              configs.LOCATIONS[configs.AREA_2].name,
                              configs.LOCATIONS[configs.AREA_3].name]
    policy_container = q_policies(available_policies, available_target_names)

    a1 = Agent(1, (0, 0, 255), initial_state, configs.AREA_1, available_policies, available_target_names,
               _obstacles=obstacles, _zones=zones, _craters=craters)
    a2 = Agent(2, (255, 0, 0), initial_state, configs.AREA_3, available_policies, available_target_names,
               _obstacles=obstacles, _zones=zones, _craters=craters)
    a3 = Agent(3, (0, 255, 0), initial_state, configs.AREA_2, available_policies, available_target_names,
               _obstacles=obstacles, _zones=zones, _craters=craters)

    rendering = MultiAgentRendering([1, 2, 3])
    rendering.change_event(new_craters=craters, new_zones=zones)

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
            rendering.state_update(a.get_state())
        rendering.render(mode='human')

        if should_stop:
            break
