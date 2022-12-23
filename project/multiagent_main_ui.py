import json
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap, QImage
import PyQt5.QtCore as QtCore
import sys
import qdarktheme
import traceback
import numpy as np

sys.path.append('../')
import project.ui_multi_agent_distractor as ui
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
            craters, zones = configs.read_scenarios(configs.SCENARIO_ID)
            self.renderer.change_event(new_craters=craters, new_zones=zones)
            self.render_map(self.renderer.render(mode="rgb_array"))

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
            self.assessment_button.clicked.connect(self.self_assessment_button_click)
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

            """
            Setup the confidence alert stuff
            """
            self.operator_alert_slider.valueChanged.connect(self.alert_slider_update)

            """
            Setup the distraction task stuff
            """
            self.report_button.clicked.connect(self.send_report)
            self.report_slider.valueChanged.connect(self.report_slider_update)
            QtCore.QTimer.singleShot(5000, self.alert_report)

        except Exception as e:
            print(e)
            traceback.print_exc()

    def state_update_robot1(self, m):
        try:
            self.robot_location.setText(str(m[configs.MultiAgentState.STATUS_LOCATION]))
            self.robot_craters.setText(str(m[configs.MultiAgentState.STATUS_HITS]))
            self.robot_zones.setText(str(m[configs.MultiAgentState.STATUS_ZONES]))
            self.robot1_cargo.setText(str(m[configs.MultiAgentState.STATUS_CARGO_COUNT]))
            color_string = "background-color: rgb({},{},{})".format(*m[configs.MultiAgentState.STATUS_COLOR])
            self.robot1_color_frame.setStyleSheet(color_string)
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
                              'zones': m[configs.MultiAgentState.STATUS_ZONES_GOA],
                              'predicted_craters': m[configs.MultiAgentState.STATUS_PREDICTED_CRATERS],
                              'predicted_zones': m[configs.MultiAgentState.STATUS_PREDICTED_ZONES],
                              'target': m[configs.MultiAgentState.STATUS_ASSESSED_GOAL]}
            self.agent1_assessment_update(assessment_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def state_update_robot2(self, m):
        try:
            self.robot2_location.setText(str(m[configs.MultiAgentState.STATUS_LOCATION]))
            self.robot2_craters.setText(str(m[configs.MultiAgentState.STATUS_HITS]))
            self.robot2_zones.setText(str(m[configs.MultiAgentState.STATUS_ZONES]))
            self.robot2_cargo.setText(str(m[configs.MultiAgentState.STATUS_CARGO_COUNT]))
            color_string = "background-color: rgb({},{},{})".format(*m[configs.MultiAgentState.STATUS_COLOR])
            self.robot2_color_frame.setStyleSheet(color_string)
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
                              'zones': m[configs.MultiAgentState.STATUS_ZONES_GOA],
                              'predicted_craters': m[configs.MultiAgentState.STATUS_PREDICTED_CRATERS],
                              'predicted_zones': m[configs.MultiAgentState.STATUS_PREDICTED_ZONES],
                              'target': m[configs.MultiAgentState.STATUS_ASSESSED_GOAL]}
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

            new_time = max(int(self.time_counter.text()), m[configs.MultiAgentState.STATUS_SIM_TIME])
            self.time_counter.setText(str(new_time))

            new_deliveries = max(int(self.delivery_counter.text()), m[configs.MultiAgentState.STATUS_DELIVERIES])
            self.delivery_counter.setText(str(new_deliveries))

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
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent1_go_area1_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT1_ID)
            self.goArea1Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT1_ID, configs.AREA_1)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent1_go_area2_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT1_ID)
            self.goArea2Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT1_ID, configs.AREA_2)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent1_go_area3_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT1_ID)
            self.goArea3Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT1_ID, configs.AREA_3)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_go_home_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT2_ID)
            self.robot2_goHomeButton.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT2_ID, configs.HOME)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_go_area1_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT2_ID)
            self.robot2_goArea1Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT2_ID, configs.AREA_1)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_go_area2_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT2_ID)
            self.robot2_goArea2Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT2_ID, configs.AREA_2)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_go_area3_button_click(self):
        try:
            self.clear_area_buttons(configs.AGENT2_ID)
            self.robot2_goArea3Button.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.goal_request(configs.AGENT2_ID, configs.AREA_3)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def end_sim_button_click(self):
        try:
            self.stopSimButton.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.end_sim()
            self.controlpub.publish(_msg)
            self.renderer.reset()
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
    def self_assessment_button_click(self):
        self.assessment_button.setStyleSheet("background-color: green")
        oa_zones = 5#self.outcome_assessment_slider_zones.value()
        oa_hits = 5#self.outcome_assessment_slider_craters.value()
        for agent_id in [configs.AGENT1_ID, configs.AGENT2_ID]:
            _msg = configs.MessageHelpers.assessment_request(agent_id, 0, oa_zones, oa_hits)
            print(_msg)
            self.controlpub.publish(_msg)

    def agent1_assessment_update(self, msg):
        """ Update the assessment display for agent 1 """
        try:
            print(msg)
            target_goal = msg['target']
            collisions = msg['collisions']
            zones = msg['zones']
            predicted_craters = msg['predicted_craters']
            predicted_zones = msg['predicted_zones']

            likelihood_string = 'Likelihood of successful navigation to {}: {}'.format(target_goal, str(collisions))
            self.robot1_likelihood_label.setText(likelihood_string)
            self.robot1_crater_assessment_value.setText(str(predicted_craters[0]) + u' \u00B1 '+str(predicted_craters[1]))
            self.robot1_zone_assessment_value.setText(str(predicted_zones[0]) + u' \u00B1 '+str(predicted_zones[1]))

        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_assessment_update(self, msg):
        """ Update the assessment display for agent 2 """
        try:
            print(msg)
            target_goal = msg['target']
            collisions = msg['collisions']
            zones = msg['zones']
            predicted_craters = msg['predicted_craters']
            predicted_zones = msg['predicted_zones']

            likelihood_string = 'Likelihood of successful navigation to {}: {}'.format(target_goal, str(collisions))
            self.robot2_likelihood_label.setText(likelihood_string)
            self.robot2_crater_assessment_value.setText(str(predicted_craters[0]) + u' \u00B1 '+str(predicted_craters[1]))
            self.robot2_zone_assessment_value.setText(str(predicted_zones[0]) + u' \u00B1 '+str(predicted_zones[1]))
        except Exception as e:
            print(e)
            traceback.print_exc()

    def render_map(self, img):
        image = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.map_area.setPixmap(pixmap)
        self.map_area.setScaledContents(True)

    def alert_slider_update(self, value):
        oa = configs.convert_famsec(float(value)/100.)
        self.outcome_threshold_value_4.setText(oa)

    def report_slider_update(self, value):
        self.report_slider_value.setText(str(value))

    def alert_report(self):
        self.report_button.setStyleSheet("background-color: {}".format('red'))
        self.report_slider_counter.setText(str(np.random.randint(0, 100)))

    def send_report(self):
        if self.report_slider_value.text() == self.report_slider_counter.text():
            self.report_button.setStyleSheet("background-color: {}".format('none'))
            QtCore.QTimer.singleShot(5000, self.alert_report)
            self.report_slider_counter.setText('-')


qdarktheme.enable_hi_dpi()

app = QApplication(sys.argv)
qdarktheme.setup_theme()
MainWindow = myMainWindow()
MainWindow.show()
app.exec_()
