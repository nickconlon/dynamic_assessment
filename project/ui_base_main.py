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
import project.ui__base.ui_base as ui
import project.communications.zmq_subscriber as subscriber
from project.communications.zmq_publisher import ZmqPublisher
import project.multiagent_configs as configs
import project.rendering_environment as rendering


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
            Setup rendering and events through the UI
            """
            self.renderer = rendering.MultiAgentRendering([configs.AGENT1_ID, configs.AGENT2_ID])
            self.render_map(self.renderer.render(mode="rgb_array"))
            self.current_event_id = 0
            self.start_time = 0
            self.sendEventButton.clicked.connect(self.send_event_button_click)
            self.startSimButton.clicked.connect(self.start_sim_button_click)
            self.time_updater_thread = QtCore.QTimer()
            self.time_updater_thread.timeout.connect(self.time_update)
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

    def temp_highlight(self, widget_func, timeout=1000, color="green"):
        widget_func.setStyleSheet("background-color: {}".format(color))
        QtCore.QTimer.singleShot(timeout, lambda: widget_func.setStyleSheet(""))

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

            if m[configs.MultiAgentState.STATUS_NEEDS_RESCUE]:
                self.renderer.previous_positions[m[configs.MultiAgentState.STATUS_AGENTID]] = []
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
        try:
            if agent_id == configs.AGENT1_ID:
                self.goHomeButton.setStyleSheet("")
                self.goArea1Button.setStyleSheet("")
                self.goArea2Button.setStyleSheet("")
                self.goArea3Button.setStyleSheet("")
            elif agent_id == configs.AGENT2_ID:
                self.robot2_goHomeButton.setStyleSheet("")
                self.robot2_goArea1Button.setStyleSheet("")
                self.robot2_goArea2Button.setStyleSheet("")
                self.robot2_goArea3Button.setStyleSheet("")
        except Exception as e:
            print(e)
            traceback.print_exc()

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
            _msg = configs.MessageHelpers.end_sim()
            self.controlpub.publish(_msg)
            self.renderer.reset()
            self.render_map(self.renderer.render(mode='rgb_array'))
            self.current_event_id = 0
            self.time_updater_thread.stop()
            self.temp_highlight(self.stopSimButton)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def time_update(self):
        self.time_counter.setText(str(int(time.time() - self.start_time)))

    def start_sim_button_click(self):
        try:
            self.start_time = time.time()
            self.time_updater_thread.start(1000)
            self.temp_highlight(self.startSimButton)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def clear_drive_mode_buttons(self, agent_id):
        try:
            if agent_id == configs.AGENT1_ID:
                self.startControlButton.setStyleSheet("")
                self.stopControlButton.setStyleSheet("")
            elif agent_id == configs.AGENT2_ID:
                self.robot2_startControlButton.setStyleSheet("")
                self.robot2_stopControlButton.setStyleSheet("")
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent1_start_control_button_click(self):
        try:
            self.clear_drive_mode_buttons(configs.AGENT1_ID)
            self.startControlButton.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.move_request(configs.AGENT1_ID, configs.MultiAgentState.START)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent1_stop_control_button_click(self):
        try:
            self.clear_drive_mode_buttons(configs.AGENT1_ID)
            self.stopControlButton.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.move_request(configs.AGENT1_ID, configs.MultiAgentState.STOP)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_start_control_button_click(self):
        try:
            self.clear_drive_mode_buttons(configs.AGENT2_ID)
            self.robot2_startControlButton.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.move_request(configs.AGENT2_ID, configs.MultiAgentState.START)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def agent2_stop_control_button_click(self):
        try:
            self.clear_drive_mode_buttons(configs.AGENT2_ID)
            self.robot2_stopControlButton.setStyleSheet("background-color: green")
            _msg = configs.MessageHelpers.move_request(configs.AGENT2_ID, configs.MultiAgentState.STOP)
            self.controlpub.publish(_msg)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def render_map(self, img):
        try:
            image = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.map_area.setPixmap(pixmap)
            self.map_area.setScaledContents(True)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def send_event_button_click(self):
        try:
            print('Sending event: ', self.current_event_id)
            craters, zones = configs.read_scenarios(self.current_event_id)
            self.renderer.change_event(new_craters=craters, new_zones=zones)
            self.render_map(self.renderer.render(mode="rgb_array"))
            event_msg = configs.MessageHelpers.pack_event(self.current_event_id)
            self.current_event_id = (self.current_event_id + 1) % 8
            self.controlpub.publish(event_msg)
            self.temp_highlight(self.sendEventButton)
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
