import copy

import time
import threading
import numpy as np
import logging
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import project.multiagent_configs as configs
from project.solvers.q_learning_policy import q_policies
from project.communications.zmq_publisher import ZmqPublisher
from project.communications.zmq_subscriber import ZmqSubscriber
from project.environment import Environment, Obstacle
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
                _topic, _msg = configs.MessageHelpers.unpack(_msg)
                if _topic == 'END_SIM':
                    self.state.needs_end_sim = True
                elif configs.MessageHelpers.for_me(_msg, self.state.agent_id):
                    if _topic == 'ASSESSMENT_REQUEST':
                        agent_id, delivery_thresh, time_thresh, collisions_thresh = configs.MessageHelpers.unpack_assessment_request(_msg)
                        self.state.delivery_assessment_threshold = delivery_thresh
                        self.state.collision_assessment_threshold = collisions_thresh
                        self.state.time_assessment_threshold = time_thresh
                        self.state.needs_assessment = True
                    elif _topic == 'MOVE_REQUEST':
                        agent_id, movement_state = configs.MessageHelpers.unpack_move_request(_msg)
                        self.state.movement_state = movement_state
                    elif _topic == 'GOAL_REQUEST':
                        agent_id, new_goal = configs.MessageHelpers.unpack_goal_request(_msg)
                        self.state.goal = new_goal
                    print('topic: {}, data: {}'.format(_topic, _msg))

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
            time.sleep(1)

    def close(self):
        self.running = False

def run_main(agent_id):
    """
        Logging
        """
    logging.basicConfig(filename='logging.log', format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("testing the log file")
    """
    Initialize zmq stuff
    """
    map_pub = ZmqPublisher("*", configs.MAP_PORT)
    control_sub = ZmqSubscriber(configs.HOST, configs.CONTROL_PORT, timeout=2500, last_only=False)
    state_pub = ZmqPublisher("*", configs.STATE_PORT)

    """
    Initialize the application state machine
    """
    robot_state = configs.MultiAgentState(agent_id, time.time(), 'red')
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
    env = Environment(robot_state.location, configs.LOCATIONS[robot_state.goal].position,
                      _obstacles=[], _zones=[], _craters=[])

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
    robot_state.current_robot_location = tuple(env.xy_from_index(state))

    """
    Self-assessment stuff
    """
    assessment = StaticAssessment()
    dynamics = DynamicAssessment()
    _, _, predicted_states, _ = assessment.rollout(policy.policies[robot_state.goal],
                                                   copy.deepcopy(env),
                                                   env.xy_from_index(state),
                                                   configs.OA_ROLLOUTS,
                                                   configs.STATE_UNCERTAINTY)
    assessment_index = 0

    """
    Publish some pre-mission data
    """
    for i in range(5):
        map_pub.publish(configs.MessageHelpers.map_out(env.render(mode="rgb_array")))
        time.sleep(1)

    """
    Live task execution
    """
    time_start = time.time()
    event_timer = 0
    print('Ready to run')
    fig, (a0, a1, a2) = plt.subplots(nrows=1, ncols=3)
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
            rewards_oa, collision_oa, times_oa = assessment.run_goa_assessment(
                policy.policies[robot_state.goal],
                copy.deepcopy(env),
                env.xy_from_index(state),
                configs.OA_ROLLOUTS,
                [robot_state.delivery_assessment_threshold,
                 robot_state.collision_assessment_threshold,
                 robot_state.time_assessment_threshold],
                [robot_state.delivery_count,
                 robot_state.collision_count,
                 robot_state.time_count],
                configs.STATE_UNCERTAINTY)

            robot_state.delivery_assessment = rewards_oa
            robot_state.time_assessment = times_oa
            robot_state.collision_assessment = collision_oa

            _, _, predicted_states, _ = assessment.rollout(policy.policies[robot_state.goal],
                                                           copy.deepcopy(env),
                                                           env.xy_from_index(state),
                                                           configs.OA_ROLLOUTS,
                                                           configs.STATE_UNCERTAINTY)
            assessment_index = 2
            robot_state.needs_assessment = False

        """
        Navigation states - get the next action
        """
        if robot_state.movement_state == configs.MultiAgentState.START:
            a = policy.policies[robot_state.goal].pi(state)
            time.sleep(configs.SPEED_AUTONOMY)

            ### TODO trying out events
            event_timer += 1
            if event_timer == 1:
                zones = [Obstacle(int(x), int(y), 1, configs.ZONE_COLOR) for (x, y) in
                         zip(np.random.normal(loc=40, scale=5, size=75),
                             np.random.normal(loc=20, scale=5, size=75))]
                craters = [Obstacle(int(x), int(y), 1, configs.CRATER_COLOR) for (x, y) in
                           zip(np.random.normal(loc=40, scale=5, size=75),
                               np.random.normal(loc=20, scale=5, size=75))]
                env.change_event(new_zones=zones, new_craters=craters)
            ###
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
            robot_state.time_count += 1
            robot_state.delivery_count += info['rewards']
            robot_state.current_robot_location = tuple(env.xy_from_index(state))

            if done:
                robot_state.needs_end_sim = True
                robot_state.movement_state = configs.MultiAgentState.STOP

            if configs.ENABLE_ET_GOA:
                needs_assessment = False
                if assessment_index >= predicted_states.shape[0]:
                    print('assessing due to no more data')
                    needs_assessment = True
                else:
                    SI = dynamics.assessment(env.xy_from_index(state)[0], predicted_states[:, assessment_index, 0], a1)
                    print('si: ', SI)
                    predicted_state_t = copy.deepcopy(predicted_states[:, assessment_index])
                    predicted_state_t = predicted_state_t[~np.isnan(predicted_state_t).any(axis=1), :]
                    p_expected = dynamics.tail(predicted_state_t, env.xy_from_index(state))

                    print('tail: ', p_expected)

                    if SI <= 0.01:
                        needs_assessment = True
                        print('assessing due to surprising data')
                assessment_index += 1
                robot_state.needs_assessment = needs_assessment

        """
        Publish the new in-mission data
        """
        map_pub.publish(configs.MessageHelpers.map_out(env.render(mode="rgb_array")))

        """
        Hold on a sec
        """
        time.sleep(0.01)

        """
        Should we end the simulation?
        """
        if robot_state.needs_end_sim:
            for _ in range(5):
                map_pub.publish(configs.MessageHelpers.map_out(env.render(mode="rgb_array")))
            break

    """
    Saving off state
    """
    print(time.time())
    print(' ', robot_state.current_robot_location)
    print(' ', a)
    print(' ', robot_state.movement_state)
    print(' ', robot_state.goal)
    print(' ', robot_state.collision_count)
    print(' ', robot_state.delivery_assessment_threshold)

    print("Ending loop and shutting down")
    thread.close()
    map_pub.close()
    control_sub.close()
    state_thread.close()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        agent_id = int(sys.argv[1])
    else:
        agent_id = configs.AGENT1_ID
    run_main(agent_id)
