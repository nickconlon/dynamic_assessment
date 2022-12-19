import json
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap, QImage
import PyQt5.QtCore as QtCore
import sys
import qdarktheme
import traceback

sys.path.append('../')
import project.ui_multi_agent as ui
import project.communications.zmq_subscriber as subscriber
from project.communications.zmq_publisher import ZmqPublisher
import project.multiagent_configs as configs


class Generic_ZeroMQ_Listener(QtCore.QObject):
    message = QtCore.pyqtSignal(str)

    def __init__(self, ip, port, topic, last_only, bind):
        QtCore.QObject.__init__(self)
        self.sub = subscriber.ZmqSubscriber(ip, port, topic, last_only=last_only, bind=bind)
        self.running = True

    def loop(self):
        while self.running:
            string = self.sub.receive()
            self.message.emit(string)


class myMainWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        try:
            QMainWindow.__init__(self)
            self.setupUi(self)
            """
            TODO Testing
            """
            import multi_agent_environment as rendering
            self.renderer = rendering.MultiAgentRendering([configs.AGENT1_ID, configs.AGENT2_ID])

            """
            Setup the control buttons
            """
            self.controlpub = ZmqPublisher("*", configs.CONTROL_PORT)
            self.startControlButton.clicked.connect(self.agent1_start_control_button_click)
            self.stopControlButton.clicked.connect(self.agent1_stop_control_button_click)
            self.robot2_startControlButton.clicked.connect(self.agent2_start_control_button_click)
            self.robot2_stopControlButton.clicked.connect(self.agent2_stop_control_button_click)

            """
            Setup the target area buttons
            """
            self.goHomeButton.clicked.connect(self.agent1_go_home_button_click)
            self.goArea1Button.clicked.connect(self.agent1_go_area1_button_click)
            self.goArea2Button.clicked.connect(self.agent1_go_area2_button_click)
            self.goArea3Button.clicked.connect(self.agent1_go_area3_button_click)

            self.robot2_goHomeButton.clicked.connect(self.agent2_go_home_button_click)
            self.robot2_goArea1Button.clicked.connect(self.agent2_go_area1_button_click)
            self.robot2_goArea2Button.clicked.connect(self.agent2_go_area2_button_click)
            self.robot2_goArea3Button.clicked.connect(self.agent2_go_area3_button_click)

            """
            End the simulation button
            """
            self.stopSimButton.clicked.connect(self.end_sim_button_click)

            """
            Setup the assessment listener thread
            """
            self.outcome_assessment_slider_rewards.valueChanged.connect(self.self_assessment_value_update)
            self.outcome_assessment_slider_zones.valueChanged.connect(self.outcome_assessment_slider_2_update)
            self.outcome_assessment_slider_craters.valueChanged.connect(self.outcome_assessment_slider_3_update)
            self.assess_button.clicked.connect(self.self_assessment_button_click)
            self.thread = QtCore.QThread()
            QtCore.QTimer.singleShot(0, self.thread.start)

            """
            Setup the state listener thread
            """
            self.stateThread = QtCore.QThread()
            self.zeromq_state_listener = Generic_ZeroMQ_Listener("*", configs.STATE_PORT, "", last_only=True, bind=True)
            self.zeromq_state_listener.moveToThread(self.stateThread)
            self.stateThread.started.connect(self.zeromq_state_listener.loop)
            self.zeromq_state_listener.message.connect(self.state_update)
            QtCore.QTimer.singleShot(0, self.stateThread.start)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def state_update_robot1(self, m):
        try:
            print(m[configs.MultiAgentState.STATUS_LOCATION])
            print(str(m[configs.MultiAgentState.STATUS_LOCATION]))
            self.robot_location.setText(str(m[configs.MultiAgentState.STATUS_LOCATION]))
            self.robot_time.setText(str(m[configs.MultiAgentState.STATUS_SIM_TIME]))
            self.robot_craters.setText(str(m[configs.MultiAgentState.STATUS_HITS]))
            self.robot_target_area.setText(str(m[configs.MultiAgentState.STATUS_GOAL]))
            self.clear_area_buttons(1)
            if m[configs.MultiAgentState.STATUS_GOAL] == configs.HOME:
                self.goHomeButton.setStyleSheet("background-color: green")
            elif m[configs.MultiAgentState.STATUS_GOAL] == configs.AREA_1:
                self.goArea1Button.setStyleSheet("background-color: green")
            elif m[configs.MultiAgentState.STATUS_GOAL] == configs.AREA_2:
                self.goArea2Button.setStyleSheet("background-color: green")
            elif m[configs.MultiAgentState.STATUS_GOAL] == configs.AREA_3:
                self.goArea3Button.setStyleSheet("background-color: green")
            self.robot_state.setText(str(m[configs.MultiAgentState.STATUS_STATE]))
            self.clear_drive_mode_buttons(configs.AGENT1_ID)
            if m[configs.MultiAgentState.STATUS_STATE] == configs.MultiAgentState.STOP:
                self.stopControlButton.setStyleSheet("background-color: green")
            elif m[configs.MultiAgentState.STATUS_STATE] == configs.MultiAgentState.START:
                self.startControlButton.setStyleSheet("background-color: green")

            assessment_msg = {'rewards': m[configs.MultiAgentState.STATUS_REWARD_GOA],
                              'collisions': m[configs.MultiAgentState.STATUS_COLLISIONS_GOA],
                              'zones': m[configs.MultiAgentState.STATUS_TIME_GOA]}
            self.agent1_assessment_update(assessment_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def state_update_robot2(self, m):
        try:
            print(m[configs.MultiAgentState.STATUS_LOCATION])
            print(str(m[configs.MultiAgentState.STATUS_LOCATION]))
            self.robot2_location.setText(str(m[configs.MultiAgentState.STATUS_LOCATION]))
            new_time = max(int(self.robot_time.text()), m[configs.MultiAgentState.STATUS_SIM_TIME])
            self.robot_time.setText(str(new_time))
            self.robot2_craters.setText(str(m[configs.MultiAgentState.STATUS_HITS]))
            self.robot2_target_area.setText(str(m[configs.MultiAgentState.STATUS_GOAL]))
            self.clear_area_buttons(configs.AGENT2_ID)
            if m[configs.MultiAgentState.STATUS_GOAL] == configs.HOME:
                self.robot2_goHomeButton.setStyleSheet("background-color: green")
            elif m[configs.MultiAgentState.STATUS_GOAL] == configs.AREA_1:
                self.robot2_goArea1Button.setStyleSheet("background-color: green")
            elif m[configs.MultiAgentState.STATUS_GOAL] == configs.AREA_2:
                self.robot2_goArea2Button.setStyleSheet("background-color: green")
            elif m[configs.MultiAgentState.STATUS_GOAL] == configs.AREA_3:
                self.robot2_goArea3Button.setStyleSheet("background-color: green")
            self.robot2_state.setText(str(m[configs.MultiAgentState.STATUS_STATE]))
            self.clear_drive_mode_buttons(configs.AGENT2_ID)
            if m[configs.MultiAgentState.STATUS_STATE] == configs.MultiAgentState.STOP:
                self.robot2_stopControlButton.setStyleSheet("background-color: green")
            elif m[configs.MultiAgentState.STATUS_STATE] == configs.MultiAgentState.START:
                self.robot2_startControlButton.setStyleSheet("background-color: green")

            assessment_msg = {'rewards': m[configs.MultiAgentState.STATUS_REWARD_GOA],
                              'collisions': m[configs.MultiAgentState.STATUS_COLLISIONS_GOA],
                              'times': m[configs.MultiAgentState.STATUS_TIME_GOA]}
            self.agent2_assessment_update(assessment_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def state_update(self, msg):
        try:
            m = json.loads(msg)
            m = m['data']
            self.renderer.state_update(m)
            img = self.renderer.render(mode="rgb_array")
            self.render_map(img)
            if m[configs.MultiAgentState.STATUS_AGENTID] == configs.AGENT1_ID:
                self.state_update_robot1(m)
            elif m[configs.MultiAgentState.STATUS_AGENTID] == configs.AGENT2_ID:
                self.state_update_robot2(m)
            else:
                print("ERROR")
        except Exception as e:
            print(e)
            traceback.print_exc()

    def clear_area_buttons(self, agent_id):
        if agent_id == configs.AGENT1_ID:
            self.goHomeButton.setStyleSheet("background-color: none")
            self.goArea1Button.setStyleSheet("background-color: none")
            self.goArea2Button.setStyleSheet("background-color: none")
            self.goArea3Button.setStyleSheet("background-color: none")
        elif agent_id == configs.AGENT2_ID:
            self.robot2_goHomeButton.setStyleSheet("background-color: none")
            self.robot2_goArea1Button.setStyleSheet("background-color: none")
            self.robot2_goArea2Button.setStyleSheet("background-color: none")
            self.robot2_goArea3Button.setStyleSheet("background-color: none")

    def agent1_go_home_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT1_ID)
            self.goHomeButton.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT1_ID, configs.HOME)
            self.controlpub.publish(_msg)
            print('home')
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent1_go_area1_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT1_ID)
            self.goArea1Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT1_ID, configs.AREA_1)
            self.controlpub.publish(_msg)
            print('area1')
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent1_go_area2_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT1_ID)
            self.goArea2Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT1_ID, configs.AREA_2)
            self.controlpub.publish(_msg)
            print('area2')
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent1_go_area3_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT1_ID)
            self.goArea3Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT1_ID, configs.AREA_3)
            self.controlpub.publish(_msg)
            print('area3')
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_go_home_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT2_ID)
            self.robot2_goHomeButton.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT2_ID, configs.HOME)
            self.controlpub.publish(_msg)
            print('home')
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_go_area1_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT2_ID)
            self.robot2_goArea1Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT2_ID, configs.AREA_1)
            self.controlpub.publish(_msg)
            print('area1')
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_go_area2_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT2_ID)
            self.robot2_goArea2Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT2_ID, configs.AREA_2)
            self.controlpub.publish(_msg)
            print('area2')
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_go_area3_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT2_ID)
            self.robot2_goArea3Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT2_ID, configs.AREA_3)
            self.controlpub.publish(_msg)
            print('area3')
        except Exception as e:
            print(e)
            traceback.print_exc()

    def end_sim_button_click(self):
        try:
            self.stopSimButton.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.end_sim()
            self.controlpub.publish(_msg)
            self.renderer.reset()
            print('stop')
        except Exception as e:
            print(e)
            traceback.print_exc()

    def clear_drive_mode_buttons(self, agent_id):
        if agent_id == configs.AGENT1_ID:
            self.startControlButton.setStyleSheet("background-color: none")
            self.stopControlButton.setStyleSheet("background-color: none")
        elif agent_id == configs.AGENT2_ID:
            self.robot2_startControlButton.setStyleSheet("background-color: none")
            self.robot2_stopControlButton.setStyleSheet("background-color: none")

    def agent1_start_control_button_click(self):
        self.clear_drive_mode_buttons(configs.AGENT1_ID)
        self.startControlButton.setStyleSheet("background-color: green")
        _msg = configs.MessageHelpers.move_request(configs.AGENT1_ID, configs.MultiAgentState.START)
        self.controlpub.publish(_msg)

    def agent1_stop_control_button_click(self):
        self.clear_drive_mode_buttons(configs.AGENT1_ID)
        self.stopControlButton.setStyleSheet("background-color: green")
        _msg = configs.MessageHelpers.move_request(configs.AGENT1_ID, configs.MultiAgentState.STOP)
        self.controlpub.publish(_msg)

    def agent2_start_control_button_click(self):
        self.clear_drive_mode_buttons(configs.AGENT2_ID)
        self.robot2_startControlButton.setStyleSheet("background-color: green")
        _msg = configs.MessageHelpers.move_request(configs.AGENT2_ID, configs.MultiAgentState.START)
        self.controlpub.publish(_msg)

    def agent2_stop_control_button_click(self):
        self.clear_drive_mode_buttons(configs.AGENT2_ID)
        self.robot2_stopControlButton.setStyleSheet("background-color: green")
        _msg = configs.MessageHelpers.move_request(configs.AGENT2_ID, configs.MultiAgentState.STOP)
        self.controlpub.publish(_msg)

    #
    # Self-assessment stuff
    #
    def self_assessment_value_update(self, value):
        self.acceptable_reward_value.setText(str(value))

    def outcome_assessment_slider_2_update(self, value):
        self.outcome_threshold_value_2.setText(str(value))

    def outcome_assessment_slider_3_update(self, value):
        self.outcome_threshold_value_3.setText(str(value))

    def self_assessment_button_click(self):
        self.assess_button.setStyleSheet("background-color: green")
        oa_rewards = self.outcome_assessment_slider_rewards.value()
        oa_zones = self.outcome_assessment_slider_zones.value()
        oa_hits = self.outcome_assessment_slider_craters.value()
        for agent_id in [configs.AGENT1_ID, configs.AGENT2_ID]:
            _msg = configs.MessageHelpers.assessment_request(agent_id, oa_rewards, oa_zones, oa_hits)
            print(_msg)
            self.controlpub.publish(_msg)

    def agent1_assessment_update(self, msg):
        """ Update the assessment display for agent 1 """
        try:
            print(msg)
            rewards = msg['rewards']
            collisions = msg['collisions']
            times = msg['zones']
            self.robot1_oa_rewards_value.setText(str(rewards))
            self.robot1_oa_rewards_value.setStyleSheet("background-color: {}".format(configs.FAMSEC_COLORS[rewards]))

            self.robot1_oa_zones_value.setText(str(times))
            self.robot1_oa_zones_value.setStyleSheet("background-color: {}".format(configs.FAMSEC_COLORS[times]))

            self.robot1_oa_hits_value.setText(str(collisions))
            self.robot1_oa_hits_value.setStyleSheet("background-color: {}".format(configs.FAMSEC_COLORS[collisions]))
            self.assess_button.setStyleSheet("background-color: none")
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_assessment_update(self, msg):
        """ Update the assessment display for agent 2 """
        try:
            print(msg)
            rewards = msg['rewards']
            collisions = msg['collisions']
            times = msg['times']

            self.robot2_oa_rewards_value.setStyleSheet("background-color: {}".format(configs.FAMSEC_COLORS[rewards]))
            self.robot2_oa_rewards_value.setText(str(times))

            self.robot2_oa_zones_value.setStyleSheet("background-color: {}".format(configs.FAMSEC_COLORS[times]))
            self.robot2_oa_zones_value.setText(str(times))

            self.robot2_oa_hits_value.setStyleSheet("background-color: {}".format(configs.FAMSEC_COLORS[collisions]))
            self.robot2_oa_hits_value.setText(str(collisions))
            self.assess_button.setStyleSheet("background-color: none")
        except Exception as e:
            print(e)
            traceback.print_exc()

    def render_map(self, img):
        image = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.map_area.setPixmap(pixmap)
        self.map_area.setScaledContents(True)


qdarktheme.enable_hi_dpi()

app = QApplication(sys.argv)
qdarktheme.setup_theme()
MainWindow = myMainWindow()
MainWindow.show()
app.exec_()
