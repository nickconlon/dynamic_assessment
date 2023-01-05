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
    def __init__(self, _subscriber, _state, _condition):
        threading.Thread.__init__(self)
        self.subscriber = _subscriber
        self.state = _state
        self.running = True
        self.condition = _condition

    def run(self):
        print("starting control thread")
        while self.running:
            _msg = self.subscriber.receive()
            if len(_msg) > 0:
                _topic, _data = configs.MessageHelpers.unpack(_msg)
                if _topic == configs.MessageHelpers.TOPICS_END_SIM:
                    self.state.needs_end_sim = True
                    logging.info(_msg)
                elif _topic == configs.MessageHelpers.TOPICS_EVENT_UPDATE:
                    event_id = configs.MessageHelpers.unpack_event(_data)
                    self.state.current_event_id = event_id
                elif _topic == configs.MessageHelpers.TOPICS_SET_ALERT:
                    new_alert_level = configs.MessageHelpers.unpack_alert_update(_data)
                    self.state.goa_alert_level = new_alert_level
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
                        if self.condition == configs.Conditions.CONDITION_ET_GOA:
                            self.state.needs_assessment = True
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
            self.state.new_assessment = False
            logging.info(_msg)
            time.sleep(0.5)

    def close(self):
        self.running = False


def run_main(_agent_id, _mission_id, _subject_id, _color,
             _transition_uncertainty, _dust_uncertainty,
             _prob_avoiding_crater, _prob_avoiding_zone, _et_threshold,
             _experimental_condition):
    print("Starting Agent:")
    print(" ID:", _agent_id)
    print(" Mission:", _mission_id)
    print(" Subject:", _subject_id)
    print(" Condition:", _experimental_condition)
    print(" Color:", _color)
    print(" p(T):", _transition_uncertainty)
    print(" p(D):", _dust_uncertainty)
    print(" p(crater avoid):", _prob_avoiding_crater)
    print(" p(zone avoid):", _prob_avoiding_zone)
    print(" ET thresh:", _et_threshold)
    logging.basicConfig(filename='{}/{}_{}_{}_{}.log'.format(configs.LOGGING_PATH, _agent_id, _mission_id, _subject_id, time.time()),
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)

    """
    Initialize zmq stuff
    """
    control_sub = ZmqSubscriber(configs.HOST, configs.CONTROL_PORT, timeout=2500, last_only=False)
    state_pub = ZmqPublisher(configs.HOST, configs.STATE_PORT, bind=False)

    """
    Initialize the application state machine
    """
    robot_state = configs.MultiAgentState(_agent_id, time.time(), configs.AGENT_COLOR[_agent_id])
    robot_state.location = configs.LOCATIONS[configs.HOME].position
    robot_state.goal = configs.LOCATIONS[configs.HOME].name
    """
    Thread for async control changes
    """
    thread = ControlThread(control_sub, robot_state, _experimental_condition)
    thread.start()

    state_thread = StateThread(state_pub, robot_state)
    state_thread.start()

    """
    Setup the environment
    """
    env = Environment(robot_state.location, configs.HOME,
                      _obstacles=[], _zones=[], _craters=[],
                      _state_transition=_transition_uncertainty,
                      _zone_transition=_dust_uncertainty,
                      _prob_avoiding_crater=_prob_avoiding_crater,
                      _prob_avoiding_zone=_prob_avoiding_zone)

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

    """
    Live task execution
    """
    print('Ready to run')
    try:
        for i in range(100000):
            """
            Make sure the env has the correct goal
            """
            if robot_state.current_event_id != -1:
                craters, zones = configs.read_scenarios(robot_state.current_event_id)
                env.change_event(new_goal_label=robot_state.goal, new_craters=craters,new_zones=zones)
            env.change_event(new_goal_label=robot_state.goal)

            """
            Run an assessment if needed
            """
            if robot_state.needs_assessment:
                print("Running the assessment!")
                assessment_report = assessment.run_another_assessment(policy.policies[robot_state.goal],
                                                                      copy.deepcopy(env),
                                                                      env.xy_from_index(state),
                                                                      configs.OA_ROLLOUTS,
                                                                      _transition_uncertainty)

                robot_state.delivery_assessment = assessment_report.delivery_goa
                robot_state.reward_assessment = 0
                robot_state.zone_assessment = 0
                robot_state.collision_assessment = 0
                robot_state.predicted_collision_count = (assessment_report.mu_craters, assessment_report.std_craters)
                robot_state.predicted_zone_count = (assessment_report.mu_zones, assessment_report.std_zones)
                robot_state.needs_assessment = False
                robot_state.new_assessment = True
                robot_state.assessed_goal = robot_state.goal
                robot_state.most_recent_report = assessment_report
                robot_state.assessment_index = 1

                label = configs.AssessmentReport.convert_famsec(robot_state.delivery_assessment)
                if configs.AssessmentReport.FAMSEC2INDEX[label] <= robot_state.goa_alert_level:
                    robot_state.movement_state = configs.MultiAgentState.STOP

            """
            Navigation states - get the next action
            """
            if robot_state.movement_state == configs.MultiAgentState.START:
                a = policy.policies[robot_state.goal].noisy_pi(state, _transition_uncertainty)
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
                        robot_state.cargo_count = min(configs.CARGO_RESUPPLY, robot_state.cargo_count+configs.CARGO_RESUPPLY)

                if robot_state.zone_count > configs.CRATERS_TO_BREAKING:
                    print("Broken :(")
                    robot_state.needs_rescue = True
                    x1 = [env.xy_from_index(state)[0], configs.LOCATIONS[configs.HOME].position[0]]
                    y1 = [env.xy_from_index(state)[1], configs.LOCATIONS[configs.HOME].position[1]]
                    from scipy.interpolate import interp1d
                    f = interp1d(x1, y1)
                    x = np.linspace(x1[0], x1[1], num=15, endpoint=True)
                    x_flipped = np.flip(x)
                    #for k in range(len(x)):
                    #    time.sleep(0.5)
                    #    robot_state.location = tuple([x_flipped[k], f(x_flipped[k])])
                    #    robot_state.color = (255, 0, 0)
                    for k in range(len(x)):
                        time.sleep(1.0)
                        robot_state.location = tuple([x[k], f(x[k])])
                        robot_state.color = (255, 0, 0)
                    state = env.reset()
                    robot_state.color = _color
                    robot_state.movement_state = configs.MultiAgentState.STOP
                    robot_state.cargo_count = configs.CARGO_RESUPPLY
                    robot_state.zone_count = 0
                    robot_state.collision_count = 0
                    robot_state.location = tuple(env.xy_from_index(state))
                    robot_state.needs_rescue = False

                if _experimental_condition == configs.Conditions.CONDITION_ET_GOA:
                    needs_assessment = False
                    if robot_state.most_recent_report is None:
                        print('assessing due to lack of data')
                        needs_assessment = True
                    elif robot_state.assessment_index >= robot_state.most_recent_report.predicted_states.shape[1]:
                        print('assessing due to lack of data')
                        needs_assessment = True
                    elif np.count_nonzero(np.isnan(robot_state.most_recent_report.predicted_states[:, robot_state.assessment_index])):
                        print('assessing due to lack of data')
                        needs_assessment = True
                    else:
                        try:
                            predicted_state_t = copy.deepcopy(robot_state.most_recent_report.predicted_states[:, robot_state.assessment_index])
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

                        if p_expected <= _et_threshold:
                            needs_assessment = True
                            print('assessing due to surprising data')
                    robot_state.assessment_index += 1
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
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        print("Ending loop and shutting down")
        thread.close()
        control_sub.close()
        state_thread.close()


if __name__ == '__main__':
    settings = configs.read_configs('ui__select/settings.yaml')

    parser = argparse.ArgumentParser(prog='test', description='')
    parser.add_argument('-a', '--agentid', type=int, default=1)
    parser.add_argument('-m', '--missionid', type=int, default=1)
    parser.add_argument('-s', '--subjectid', type=int, default=1)
    parser.add_argument('-e', '--et_thresh',  type=float, default=configs.ET_THRESHOLD)
    parser.add_argument('-t', '--p_transition', type=float, default=configs.STATE_UNCERTAINTY)
    parser.add_argument('-d', '--p_dust', type=float, default=configs.ZONE_UNCERTAINTY)
    args = parser.parse_args()

    agent_id = args.agentid
    mission_id = args.missionid
    subject_id = args.subjectid
    agent_et_threshold = args.et_thresh
    agent_color = configs.AGENT_COLOR[agent_id]
    agent_transition_uncertainty = args.p_transition
    agent_dust_uncertainty = args.p_dust
    run_main(agent_id, mission_id, subject_id,
             agent_color,
             agent_transition_uncertainty,
             agent_dust_uncertainty,
             agent_et_threshold)