import copy

import time
import threading
import traceback
import logging
import numpy as np
import sys
import argparse

sys.path.append('../')
import project.multiagent_configs as configs
from project.solvers.q_learning_policy import q_policies
from project.communications.zmq_publisher import ZmqPublisher
from project.communications.zmq_subscriber import ZmqSubscriber
from project.environment import Environment
from project.assessment.assessments import StaticAssessment, DynamicAssessment


class ControlThread(threading.Thread):
    def __init__(self, _subscriber, _state):
        threading.Thread.__init__(self)
        self.subscriber = _subscriber
        self.state = _state
        self.running = True

    def run(self):
        print("starting control thread")
        while self.running:
            _msg = self.subscriber.receive()
            if len(_msg) > 0:
                _topic, _data = configs.MessageHelpers.unpack(_msg)
                if _topic == configs.MessageHelpers.TOPICS_END_SIM:
                    self.state.needs_end_sim = True
                    logging.info(_msg)
                elif configs.MessageHelpers.for_me(_data, self.state.agent_id):
                    if _topic == configs.MessageHelpers.TOPICS_ASSESSMENT_REQUEST:
                        agent_id, delivery_thresh, time_thresh, collisions_thresh = configs.MessageHelpers.unpack_assessment_request(_data)
                        self.state.reward_assessment_threshold = delivery_thresh
                        self.state.collision_assessment_threshold = collisions_thresh
                        self.state.zone_assessment_threshold = time_thresh
                        self.state.needs_assessment = True
                    elif _topic == configs.MessageHelpers.TOPICS_MOVE_REQUEST:
                        agent_id, movement_state = configs.MessageHelpers.unpack_move_request(_data)
                        self.state.movement_state = movement_state
                    elif _topic == configs.MessageHelpers.TOPICS_GOAL_REQUEST:
                        agent_id, new_goal = configs.MessageHelpers.unpack_goal_request(_data)
                        self.state.goal = new_goal
                        self.state.at_goal = False

                    logging.info(_msg)
                    print('topic: {}, data: {}'.format(_topic, _data))

    def close(self):
        self.running = False


class StateThread(threading.Thread):
    def __init__(self, _publisher, _state):
        threading.Thread.__init__(self)
        self.publisher = _publisher
        self.state = _state
        self.running = True

    def run(self):
        while self.running:
            self.state.sim_current_time = int(time.time() - self.state.sim_start_time)
            _msg = configs.MessageHelpers.state_update(self.state.state_update_message())
            self.publisher.publish(_msg)
            logging.info(_msg)
            time.sleep(0.5)

    def close(self):
        self.running = False


def run_main(agent_id, mission_id, subject_id):
    logging.basicConfig(filename='{}/{}_{}_{}_{}.log'.format(configs.LOGGING_PATH, agent_id, mission_id, subject_id, time.time()),
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)

    """
    Initialize zmq stuff
    """
    map_pub = ZmqPublisher(configs.HOST, configs.MAP_PORT, bind=False)
    control_sub = ZmqSubscriber(configs.HOST, configs.CONTROL_PORT, timeout=2500, last_only=False)
    state_pub = ZmqPublisher(configs.HOST, configs.STATE_PORT, bind=False)

    """
    Initialize the application state machine
    """
    robot_state = configs.MultiAgentState(agent_id, time.time(), configs.AGENT_COLOR[agent_id])
    robot_state.location = configs.LOCATIONS[configs.HOME].position
    robot_state.goal = configs.LOCATIONS[configs.HOME].name
    """
    Thread for async control changes
    """
    thread = ControlThread(control_sub, robot_state)
    thread.start()

    state_thread = StateThread(state_pub, robot_state)
    state_thread.start()

    """
    Setup the environment
    """
    env = Environment(robot_state.location, configs.HOME, _obstacles=[], _zones=[], _craters=[])
    craters, zones = configs.read_scenarios(configs.SCENARIO_ID)
    env.change_event(new_craters=craters, new_zones=zones)

    """
    Setup the control policies
    """
    available_policies = [configs.LOCATIONS[configs.HOME].policy,
                          configs.LOCATIONS[configs.AREA_1].policy,
                          configs.LOCATIONS[configs.AREA_2].policy,
                          configs.LOCATIONS[configs.AREA_3].policy]
    available_target_names = [configs.LOCATIONS[configs.HOME].name,
                              configs.LOCATIONS[configs.AREA_1].name,
                              configs.LOCATIONS[configs.AREA_2].name,
                              configs.LOCATIONS[configs.AREA_3].name]
    policy = q_policies(available_policies, available_target_names)

    """
    Initial state
    """
    state = env.reset()
    robot_state.location = tuple(env.xy_from_index(state))

    """
    Self-assessment stuff
    """
    assessment = StaticAssessment()
    dynamics = DynamicAssessment()
    assessment_index = 0
    predicted_states = np.array([[robot_state.location]])

    """
    Live task execution
    """
    print('Ready to run')
    for i in range(10000):
        """
        Make sure the env has the correct goal
        """
        env.change_event(new_goal_label=robot_state.goal)

        """
        Run an assessment if needed
        """
        if robot_state.needs_assessment:
            print("Running the assessment!")
            c_oa, c_m, c_std, z_oa, z_m, z_std, predicted_states = assessment.run_goa_assessment_new(
                policy.policies[robot_state.goal],
                copy.deepcopy(env),
                env.xy_from_index(state),
                configs.OA_ROLLOUTS,
                [robot_state.reward_assessment_threshold,
                 robot_state.collision_assessment_threshold,
                 robot_state.zone_assessment_threshold],
                [0, 0,0],
                configs.STATE_UNCERTAINTY)

            robot_state.reward_assessment = 0
            robot_state.zone_assessment = z_oa
            robot_state.collision_assessment = c_oa

            robot_state.predicted_collision_count = (np.ceil(c_m), np.ceil(c_std))
            robot_state.predicted_zone_count = (np.ceil(z_m), np.ceil(z_std))

            assessment_index = 1
            robot_state.needs_assessment = False
            robot_state.assessed_goal = robot_state.goal

            if c_oa <= 0.4:
                robot_state.movement_state = configs.MultiAgentState.STOP

        """
        Navigation states - get the next action
        """
        if robot_state.movement_state == configs.MultiAgentState.START:
            a = policy.policies[robot_state.goal].pi(state)
            time.sleep(configs.SPEED_AUTONOMY)
        else:
            a = -1

        """
        Execute the next action
        """
        if a in env.action_space:
            state, reward, done, info = env.step(a)
            """
            Update the current state
            """
            robot_state.collision_count += info['collisions']
            robot_state.zone_count += info['zones']
            robot_state.reward_count += info['rewards']

            robot_state.location = tuple(env.xy_from_index(state))

            if done:
                robot_state.movement_state = configs.MultiAgentState.STOP
                robot_state.at_goal = True
                if info['location'] != configs.HOME:
                    if robot_state.cargo_count >= 1:
                        robot_state.delivery_count += 1
                        robot_state.cargo_count -= 1
                elif info['location'] == configs.HOME:
                    robot_state.cargo_count = min(3, robot_state.cargo_count+3)

            if configs.ENABLE_ET_GOA:
                needs_assessment = False
                if assessment_index >= predicted_states.shape[1] or np.count_nonzero(np.isnan(predicted_states[:, assessment_index])):
                    print('assessing due to lack of data')
                    needs_assessment = True
                else:
                    try:
                        predicted_state_t = copy.deepcopy(predicted_states[:, assessment_index])
                        predicted_state_t = predicted_state_t[~np.isnan(predicted_state_t).any(axis=1), :]
                        p_expected = dynamics.tail(predicted_state_t, env.xy_from_index(state))
                        print("tail: {:.2f}".format(p_expected))
                    except Exception as e:
                        p_expected = 0.0
                        print(e)
                        traceback.print_exc()
                    try:
                        # TODO map changes SI
                        pass
                    except Exception as e:
                        p_expected = 0.0
                        print(e)
                        traceback.print_exc()

                    if p_expected <= 0.01:
                        needs_assessment = True
                        print('assessing due to surprising data')
                assessment_index += 1
                robot_state.needs_assessment = needs_assessment

        """
        Hold on a sec
        """
        time.sleep(0.01)

        """
        Should we end the simulation?
        """
        if robot_state.needs_end_sim:
            break

    """
    Saving off state
    """
    print(time.time())
    print('location: ', robot_state.location)
    print('state: ', robot_state.movement_state)
    print('goal: ', robot_state.goal)
    print('craters: ', robot_state.collision_count)
    print('zones: ', robot_state.zone_count)
    print('reward: ', robot_state.reward_count)
    print('deliveries: ', robot_state.delivery_count)

    print("Ending loop and shutting down")
    thread.close()
    map_pub.close()
    control_sub.close()
    state_thread.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='test', description='')
    parser.add_argument('-a', '--agentid', required=True, type=int)
    #parser.add_argument('-m', '--missionid', required=True, type=int)
    #parser.add_argument('-s', '--subjectid', required=True, type=int)
    args = parser.parse_args()

    agent_id = args.agentid
    mission_id = 1#args.missionid
    subject_id = 1#args.subjectid
    run_main(agent_id, mission_id, subject_id)
