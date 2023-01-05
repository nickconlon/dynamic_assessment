import json
import time

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap, QImage
import PyQt5.QtCore as QtCore
import sys
import qdarktheme
import traceback
import numpy as np

sys.path.append('../')
import project.ui_select_assessment as ui
import project.communications.zmq_subscriber as subscriber
from project.communications.zmq_publisher import ZmqPublisher
import project.multiagent_configs as configs
import multi_agent_environment as rendering


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
            Setup the control buttons
            """
            self.controlpub = ZmqPublisher("localhost", configs.CONTROL_PORT, bind=False)

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
            self.zeromq_state_listener = Generic_ZeroMQ_Listener("localhost", configs.STATE_PORT, "", last_only=True, bind=False)
            self.zeromq_state_listener.moveToThread(self.stateThread)
            self.stateThread.started.connect(self.zeromq_state_listener.loop)
            self.zeromq_state_listener.message.connect(self.state_update)
            QtCore.QTimer.singleShot(0, self.stateThread.start)


        except Exception as e:
            print(e)
            traceback.print_exc()

    def temp_highlight(self, widget_func, timeout=1000, color="green"):
        widget_func.setStyleSheet("background-color: {}".format(color))
        QtCore.QTimer.singleShot(timeout, lambda: widget_func.setStyleSheet(""))

    def state_update_robot1(self, m):
        try:
            if m[configs.MultiAgentState.STATUS_NEW_ASSESSMENT]:
                assessment_msg = {'rewards': m[configs.MultiAgentState.STATUS_REWARD_GOA],
                                  'collisions': m[configs.MultiAgentState.STATUS_COLLISIONS_GOA],
                                  'zones': m[configs.MultiAgentState.STATUS_ZONES_GOA],
                                  'predicted_craters': m[configs.MultiAgentState.STATUS_PREDICTED_CRATERS],
                                  'predicted_zones': m[configs.MultiAgentState.STATUS_PREDICTED_ZONES],
                                  'target': m[configs.MultiAgentState.STATUS_ASSESSED_GOAL],
                                  'deliveries': m[configs.MultiAgentState.STATUS_DELIVERIES_GOA]}
                self.agent1_assessment_update(assessment_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def state_update_robot2(self, m):
        try:
            if m[configs.MultiAgentState.STATUS_NEW_ASSESSMENT]:
                assessment_msg = {'rewards': m[configs.MultiAgentState.STATUS_REWARD_GOA],
                                  'collisions': m[configs.MultiAgentState.STATUS_COLLISIONS_GOA],
                                  'zones': m[configs.MultiAgentState.STATUS_ZONES_GOA],
                                  'predicted_craters': m[configs.MultiAgentState.STATUS_PREDICTED_CRATERS],
                                  'predicted_zones': m[configs.MultiAgentState.STATUS_PREDICTED_ZONES],
                                  'target': m[configs.MultiAgentState.STATUS_ASSESSED_GOAL],
                                  'deliveries': m[configs.MultiAgentState.STATUS_DELIVERIES_GOA]}
                self.agent2_assessment_update(assessment_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def state_update(self, msg):
        try:
            m = json.loads(msg)
            m = m['data']
            if m[configs.MultiAgentState.STATUS_AGENTID] == configs.AGENT1_ID:
                self.state_update_robot1(m)
            elif m[configs.MultiAgentState.STATUS_AGENTID] == configs.AGENT2_ID:
                self.state_update_robot2(m)
            else:
                print("ERROR")
        except Exception as e:
            print(e)
            traceback.print_exc()

    #
    # Self-assessment stuff
    #
    def self_assessment_button_click(self):
        try:
            self.assessment_button.setStyleSheet("background-color: green")
            oa_zones = 5
            oa_hits = 5
            for agent_id in [configs.AGENT1_ID, configs.AGENT2_ID]:
                _msg = configs.MessageHelpers.assessment_request(agent_id, 0, oa_zones, oa_hits)
                print(_msg)
                self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def clear_assessment_button(self):
        self.assessment_button.setStyleSheet("")

    def agent1_assessment_update(self, msg):
        """ Update the assessment display for agent 1 """
        try:
            self.clear_assessment_button()
            target_goal = msg['target']
            collisions = msg['collisions']
            zones = msg['zones']
            predicted_craters = msg['predicted_craters']
            predicted_zones = msg['predicted_zones']
            deliveries_goa = msg['deliveries']

            likelihood_string = 'Likelihood of successful navigation to {}: {}'.format(target_goal, str(deliveries_goa))
            self.robot1_likelihood_label.setText(likelihood_string)
            self.robot1_crater_assessment_value.setText(str(predicted_craters[0]) + u' \u00B1 '+str(predicted_craters[1]))
            self.robot1_zone_assessment_value.setText(str(predicted_zones[0]) + u' \u00B1 '+str(predicted_zones[1]))
            self.temp_highlight(self.frame_10)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_assessment_update(self, msg):
        """ Update the assessment display for agent 2 """
        try:
            self.clear_assessment_button()
            target_goal = msg['target']
            collisions = msg['collisions']
            zones = msg['zones']
            predicted_craters = msg['predicted_craters']
            predicted_zones = msg['predicted_zones']
            deliveries_goa = msg['deliveries']

            likelihood_string = 'Likelihood of successful navigation to {}: {}'.format(target_goal, str(deliveries_goa))
            self.robot2_likelihood_label.setText(likelihood_string)
            self.robot2_crater_assessment_value.setText(str(predicted_craters[0]) + u' \u00B1 '+str(predicted_craters[1]))
            self.robot2_zone_assessment_value.setText(str(predicted_zones[0]) + u' \u00B1 '+str(predicted_zones[1]))
            self.temp_highlight(self.frame_11)
        except Exception as e:
            print(e)
            traceback.print_exc()


if __name__ == '__main__':
    try:
        qdarktheme.enable_hi_dpi()
        app = QApplication(sys.argv)
        qdarktheme.setup_theme()
        MainWindow = myMainWindow()
        MainWindow.show()
        app.exec_()
    except Exception as e:
        print(e)
        traceback.print_exc()
