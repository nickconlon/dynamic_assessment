# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_multi_agent_distractor.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1542, 747)
        font = QtGui.QFont()
        font.setPointSize(18)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(580, 10, 551, 491))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_4.setLineWidth(4)
        self.frame_4.setObjectName("frame_4")
        self.time_description_label = QtWidgets.QLabel(self.frame_4)
        self.time_description_label.setGeometry(QtCore.QRect(10, 10, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.time_description_label.setFont(font)
        self.time_description_label.setObjectName("time_description_label")
        self.time_counter = QtWidgets.QLabel(self.frame_4)
        self.time_counter.setGeometry(QtCore.QRect(140, 10, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.time_counter.setFont(font)
        self.time_counter.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.time_counter.setObjectName("time_counter")
        self.task_assessment_label_2 = QtWidgets.QLabel(self.frame_4)
        self.task_assessment_label_2.setGeometry(QtCore.QRect(100, 130, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.task_assessment_label_2.setFont(font)
        self.task_assessment_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.task_assessment_label_2.setObjectName("task_assessment_label_2")
        self.task_assessment_label_3 = QtWidgets.QLabel(self.frame_4)
        self.task_assessment_label_3.setGeometry(QtCore.QRect(310, 130, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.task_assessment_label_3.setFont(font)
        self.task_assessment_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.task_assessment_label_3.setObjectName("task_assessment_label_3")
        self.agent2_frame = QtWidgets.QFrame(self.frame_4)
        self.agent2_frame.setGeometry(QtCore.QRect(10, 320, 531, 141))
        self.agent2_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.agent2_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.agent2_frame.setObjectName("agent2_frame")
        self.frame_13 = QtWidgets.QFrame(self.agent2_frame)
        self.frame_13.setGeometry(QtCore.QRect(90, 10, 201, 121))
        self.frame_13.setAutoFillBackground(False)
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_13.setLineWidth(4)
        self.frame_13.setObjectName("frame_13")
        self.robot2_zones = QtWidgets.QLabel(self.frame_13)
        self.robot2_zones.setGeometry(QtCore.QRect(110, 90, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot2_zones.setFont(font)
        self.robot2_zones.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot2_zones.setObjectName("robot2_zones")
        self.robot2_location = QtWidgets.QLabel(self.frame_13)
        self.robot2_location.setGeometry(QtCore.QRect(110, 20, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot2_location.setFont(font)
        self.robot2_location.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot2_location.setObjectName("robot2_location")
        self.robot2_state = QtWidgets.QLabel(self.frame_13)
        self.robot2_state.setGeometry(QtCore.QRect(110, 0, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot2_state.setFont(font)
        self.robot2_state.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot2_state.setObjectName("robot2_state")
        self.location_description_label_3 = QtWidgets.QLabel(self.frame_13)
        self.location_description_label_3.setGeometry(QtCore.QRect(10, 20, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.location_description_label_3.setFont(font)
        self.location_description_label_3.setObjectName("location_description_label_3")
        self.robot2_craters = QtWidgets.QLabel(self.frame_13)
        self.robot2_craters.setGeometry(QtCore.QRect(110, 70, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot2_craters.setFont(font)
        self.robot2_craters.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot2_craters.setObjectName("robot2_craters")
        self.state_description_label_3 = QtWidgets.QLabel(self.frame_13)
        self.state_description_label_3.setGeometry(QtCore.QRect(10, 0, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.state_description_label_3.setFont(font)
        self.state_description_label_3.setObjectName("state_description_label_3")
        self.hits_description_label_3 = QtWidgets.QLabel(self.frame_13)
        self.hits_description_label_3.setGeometry(QtCore.QRect(10, 70, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.hits_description_label_3.setFont(font)
        self.hits_description_label_3.setObjectName("hits_description_label_3")
        self.area_description_label_3 = QtWidgets.QLabel(self.frame_13)
        self.area_description_label_3.setGeometry(QtCore.QRect(10, 90, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.area_description_label_3.setFont(font)
        self.area_description_label_3.setObjectName("area_description_label_3")
        self.robot2_cargo_description_label = QtWidgets.QLabel(self.frame_13)
        self.robot2_cargo_description_label.setGeometry(QtCore.QRect(10, 40, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.robot2_cargo_description_label.setFont(font)
        self.robot2_cargo_description_label.setObjectName("robot2_cargo_description_label")
        self.robot2_cargo = QtWidgets.QLabel(self.frame_13)
        self.robot2_cargo.setGeometry(QtCore.QRect(110, 40, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot2_cargo.setFont(font)
        self.robot2_cargo.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot2_cargo.setObjectName("robot2_cargo")
        self.frame_14 = QtWidgets.QFrame(self.agent2_frame)
        self.frame_14.setGeometry(QtCore.QRect(300, 10, 221, 121))
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_14.setLineWidth(4)
        self.frame_14.setObjectName("frame_14")
        self.robot2_goArea2Button = QtWidgets.QPushButton(self.frame_14)
        self.robot2_goArea2Button.setGeometry(QtCore.QRect(10, 80, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.robot2_goArea2Button.setFont(font)
        self.robot2_goArea2Button.setObjectName("robot2_goArea2Button")
        self.robot2_goHomeButton = QtWidgets.QPushButton(self.frame_14)
        self.robot2_goHomeButton.setGeometry(QtCore.QRect(10, 50, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.robot2_goHomeButton.setFont(font)
        self.robot2_goHomeButton.setObjectName("robot2_goHomeButton")
        self.robot2_goArea1Button = QtWidgets.QPushButton(self.frame_14)
        self.robot2_goArea1Button.setGeometry(QtCore.QRect(110, 50, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.robot2_goArea1Button.setFont(font)
        self.robot2_goArea1Button.setObjectName("robot2_goArea1Button")
        self.robot2_goArea3Button = QtWidgets.QPushButton(self.frame_14)
        self.robot2_goArea3Button.setGeometry(QtCore.QRect(110, 80, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.robot2_goArea3Button.setFont(font)
        self.robot2_goArea3Button.setObjectName("robot2_goArea3Button")
        self.robot2_startControlButton = QtWidgets.QPushButton(self.frame_14)
        self.robot2_startControlButton.setGeometry(QtCore.QRect(20, 10, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.robot2_startControlButton.setFont(font)
        self.robot2_startControlButton.setFlat(False)
        self.robot2_startControlButton.setObjectName("robot2_startControlButton")
        self.robot2_stopControlButton = QtWidgets.QPushButton(self.frame_14)
        self.robot2_stopControlButton.setGeometry(QtCore.QRect(110, 10, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.robot2_stopControlButton.setFont(font)
        self.robot2_stopControlButton.setObjectName("robot2_stopControlButton")
        self.robot2_color_frame = QtWidgets.QFrame(self.agent2_frame)
        self.robot2_color_frame.setGeometry(QtCore.QRect(10, 10, 71, 121))
        self.robot2_color_frame.setStyleSheet("")
        self.robot2_color_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.robot2_color_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.robot2_color_frame.setObjectName("robot2_color_frame")
        self.agent1_frame = QtWidgets.QFrame(self.frame_4)
        self.agent1_frame.setGeometry(QtCore.QRect(10, 170, 531, 141))
        self.agent1_frame.setAutoFillBackground(False)
        self.agent1_frame.setStyleSheet("")
        self.agent1_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.agent1_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.agent1_frame.setObjectName("agent1_frame")
        self.frame = QtWidgets.QFrame(self.agent1_frame)
        self.frame.setGeometry(QtCore.QRect(90, 10, 201, 121))
        self.frame.setAutoFillBackground(False)
        self.frame.setStyleSheet("")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setLineWidth(4)
        self.frame.setObjectName("frame")
        self.robot_zones = QtWidgets.QLabel(self.frame)
        self.robot_zones.setGeometry(QtCore.QRect(110, 90, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot_zones.setFont(font)
        self.robot_zones.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot_zones.setObjectName("robot_zones")
        self.robot_location = QtWidgets.QLabel(self.frame)
        self.robot_location.setGeometry(QtCore.QRect(110, 20, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot_location.setFont(font)
        self.robot_location.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot_location.setObjectName("robot_location")
        self.robot_state = QtWidgets.QLabel(self.frame)
        self.robot_state.setGeometry(QtCore.QRect(110, 0, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot_state.setFont(font)
        self.robot_state.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot_state.setObjectName("robot_state")
        self.location_description_label = QtWidgets.QLabel(self.frame)
        self.location_description_label.setGeometry(QtCore.QRect(10, 20, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.location_description_label.setFont(font)
        self.location_description_label.setObjectName("location_description_label")
        self.robot_craters = QtWidgets.QLabel(self.frame)
        self.robot_craters.setGeometry(QtCore.QRect(110, 70, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot_craters.setFont(font)
        self.robot_craters.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot_craters.setObjectName("robot_craters")
        self.state_description_label = QtWidgets.QLabel(self.frame)
        self.state_description_label.setGeometry(QtCore.QRect(10, 0, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.state_description_label.setFont(font)
        self.state_description_label.setObjectName("state_description_label")
        self.hits_description_label = QtWidgets.QLabel(self.frame)
        self.hits_description_label.setGeometry(QtCore.QRect(10, 70, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.hits_description_label.setFont(font)
        self.hits_description_label.setObjectName("hits_description_label")
        self.zones_description_label = QtWidgets.QLabel(self.frame)
        self.zones_description_label.setGeometry(QtCore.QRect(10, 90, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.zones_description_label.setFont(font)
        self.zones_description_label.setObjectName("zones_description_label")
        self.cargo_description_label = QtWidgets.QLabel(self.frame)
        self.cargo_description_label.setGeometry(QtCore.QRect(10, 40, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.cargo_description_label.setFont(font)
        self.cargo_description_label.setObjectName("cargo_description_label")
        self.robot1_cargo = QtWidgets.QLabel(self.frame)
        self.robot1_cargo.setGeometry(QtCore.QRect(110, 40, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.robot1_cargo.setFont(font)
        self.robot1_cargo.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.robot1_cargo.setObjectName("robot1_cargo")
        self.frame_8 = QtWidgets.QFrame(self.agent1_frame)
        self.frame_8.setGeometry(QtCore.QRect(300, 10, 221, 121))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_8.setLineWidth(4)
        self.frame_8.setObjectName("frame_8")
        self.goArea2Button = QtWidgets.QPushButton(self.frame_8)
        self.goArea2Button.setGeometry(QtCore.QRect(10, 80, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.goArea2Button.setFont(font)
        self.goArea2Button.setObjectName("goArea2Button")
        self.goHomeButton = QtWidgets.QPushButton(self.frame_8)
        self.goHomeButton.setGeometry(QtCore.QRect(10, 50, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.goHomeButton.setFont(font)
        self.goHomeButton.setObjectName("goHomeButton")
        self.goArea1Button = QtWidgets.QPushButton(self.frame_8)
        self.goArea1Button.setGeometry(QtCore.QRect(110, 50, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.goArea1Button.setFont(font)
        self.goArea1Button.setObjectName("goArea1Button")
        self.goArea3Button = QtWidgets.QPushButton(self.frame_8)
        self.goArea3Button.setGeometry(QtCore.QRect(110, 80, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.goArea3Button.setFont(font)
        self.goArea3Button.setObjectName("goArea3Button")
        self.startControlButton = QtWidgets.QPushButton(self.frame_8)
        self.startControlButton.setGeometry(QtCore.QRect(20, 10, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.startControlButton.setFont(font)
        self.startControlButton.setFlat(False)
        self.startControlButton.setObjectName("startControlButton")
        self.stopControlButton = QtWidgets.QPushButton(self.frame_8)
        self.stopControlButton.setGeometry(QtCore.QRect(110, 10, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.stopControlButton.setFont(font)
        self.stopControlButton.setObjectName("stopControlButton")
        self.robot1_color_frame = QtWidgets.QFrame(self.agent1_frame)
        self.robot1_color_frame.setGeometry(QtCore.QRect(10, 10, 71, 121))
        self.robot1_color_frame.setStyleSheet("")
        self.robot1_color_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.robot1_color_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.robot1_color_frame.setObjectName("robot1_color_frame")
        self.delivery_description_label = QtWidgets.QLabel(self.frame_4)
        self.delivery_description_label.setGeometry(QtCore.QRect(10, 40, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.delivery_description_label.setFont(font)
        self.delivery_description_label.setObjectName("delivery_description_label")
        self.delivery_counter = QtWidgets.QLabel(self.frame_4)
        self.delivery_counter.setGeometry(QtCore.QRect(140, 40, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.delivery_counter.setFont(font)
        self.delivery_counter.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.delivery_counter.setObjectName("delivery_counter")
        self.task_assessment_label_4 = QtWidgets.QLabel(self.frame_4)
        self.task_assessment_label_4.setGeometry(QtCore.QRect(10, 130, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.task_assessment_label_4.setFont(font)
        self.task_assessment_label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.task_assessment_label_4.setObjectName("task_assessment_label_4")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(20, 480, 551, 61))
        self.frame_3.setAutoFillBackground(False)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_3.setLineWidth(4)
        self.frame_3.setObjectName("frame_3")
        self.stopSimButton = QtWidgets.QPushButton(self.frame_3)
        self.stopSimButton.setGeometry(QtCore.QRect(10, 10, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.stopSimButton.setFont(font)
        self.stopSimButton.setObjectName("stopSimButton")
        self.sendEventButton = QtWidgets.QPushButton(self.frame_3)
        self.sendEventButton.setGeometry(QtCore.QRect(280, 10, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.sendEventButton.setFont(font)
        self.sendEventButton.setObjectName("sendEventButton")
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(20, 10, 551, 461))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_5.setLineWidth(4)
        self.frame_5.setObjectName("frame_5")
        self.map_area = QtWidgets.QLabel(self.frame_5)
        self.map_area.setGeometry(QtCore.QRect(10, 10, 531, 401))
        self.map_area.setObjectName("map_area")
        self.dust_zone_frame = QtWidgets.QFrame(self.frame_5)
        self.dust_zone_frame.setGeometry(QtCore.QRect(10, 420, 31, 31))
        self.dust_zone_frame.setStyleSheet("background-color: rgb(105, 57, 116)")
        self.dust_zone_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.dust_zone_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.dust_zone_frame.setObjectName("dust_zone_frame")
        self.dust_zone_color_label = QtWidgets.QLabel(self.frame_5)
        self.dust_zone_color_label.setGeometry(QtCore.QRect(50, 420, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dust_zone_color_label.setFont(font)
        self.dust_zone_color_label.setObjectName("dust_zone_color_label")
        self.crater_zone_frame = QtWidgets.QFrame(self.frame_5)
        self.crater_zone_frame.setGeometry(QtCore.QRect(200, 420, 31, 31))
        self.crater_zone_frame.setStyleSheet("background-color:rgb(162, 48, 65)")
        self.crater_zone_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.crater_zone_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.crater_zone_frame.setObjectName("crater_zone_frame")
        self.crater_zone_color_label = QtWidgets.QLabel(self.frame_5)
        self.crater_zone_color_label.setGeometry(QtCore.QRect(240, 420, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.crater_zone_color_label.setFont(font)
        self.crater_zone_color_label.setObjectName("crater_zone_color_label")
        self.blocke_zone_frame = QtWidgets.QFrame(self.frame_5)
        self.blocke_zone_frame.setGeometry(QtCore.QRect(400, 420, 31, 31))
        self.blocke_zone_frame.setStyleSheet("background-color:rgb(112, 112, 112)")
        self.blocke_zone_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.blocke_zone_frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.blocke_zone_frame.setObjectName("blocke_zone_frame")
        self.blocked_zone_color_label = QtWidgets.QLabel(self.frame_5)
        self.blocked_zone_color_label.setGeometry(QtCore.QRect(440, 420, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.blocked_zone_color_label.setFont(font)
        self.blocked_zone_color_label.setObjectName("blocked_zone_color_label")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(20, 550, 1031, 101))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setObjectName("frame_2")
        self.report_button = QtWidgets.QPushButton(self.frame_2)
        self.report_button.setGeometry(QtCore.QRect(10, 10, 161, 81))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.report_button.setFont(font)
        self.report_button.setObjectName("report_button")
        self.report_slider = QtWidgets.QSlider(self.frame_2)
        self.report_slider.setGeometry(QtCore.QRect(300, 60, 721, 22))
        self.report_slider.setOrientation(QtCore.Qt.Horizontal)
        self.report_slider.setInvertedAppearance(False)
        self.report_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.report_slider.setTickInterval(10)
        self.report_slider.setObjectName("report_slider")
        self.report_slider_counter = QtWidgets.QLabel(self.frame_2)
        self.report_slider_counter.setGeometry(QtCore.QRect(186, 20, 91, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.report_slider_counter.setFont(font)
        self.report_slider_counter.setAlignment(QtCore.Qt.AlignCenter)
        self.report_slider_counter.setObjectName("report_slider_counter")
        self.report_slider_value = QtWidgets.QLabel(self.frame_2)
        self.report_slider_value.setGeometry(QtCore.QRect(600, 10, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.report_slider_value.setFont(font)
        self.report_slider_value.setAlignment(QtCore.Qt.AlignCenter)
        self.report_slider_value.setObjectName("report_slider_value")
        self.frame_6 = QtWidgets.QFrame(self.centralwidget)
        self.frame_6.setGeometry(QtCore.QRect(1140, 10, 381, 491))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_6.setObjectName("frame_6")
        self.frame_11 = QtWidgets.QFrame(self.frame_6)
        self.frame_11.setGeometry(QtCore.QRect(10, 330, 361, 121))
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_11.setObjectName("frame_11")
        self.label_11 = QtWidgets.QLabel(self.frame_11)
        self.label_11.setGeometry(QtCore.QRect(10, 0, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.robot2_likelihood_label = QtWidgets.QLabel(self.frame_11)
        self.robot2_likelihood_label.setGeometry(QtCore.QRect(10, 30, 351, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.robot2_likelihood_label.setFont(font)
        self.robot2_likelihood_label.setObjectName("robot2_likelihood_label")
        self.label_13 = QtWidgets.QLabel(self.frame_11)
        self.label_13.setGeometry(QtCore.QRect(10, 50, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.robot2_crater_assessment_value = QtWidgets.QLabel(self.frame_11)
        self.robot2_crater_assessment_value.setGeometry(QtCore.QRect(80, 70, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.robot2_crater_assessment_value.setFont(font)
        self.robot2_crater_assessment_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.robot2_crater_assessment_value.setObjectName("robot2_crater_assessment_value")
        self.robot2_zone_assessment_value = QtWidgets.QLabel(self.frame_11)
        self.robot2_zone_assessment_value.setGeometry(QtCore.QRect(80, 90, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.robot2_zone_assessment_value.setFont(font)
        self.robot2_zone_assessment_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.robot2_zone_assessment_value.setObjectName("robot2_zone_assessment_value")
        self.label_14 = QtWidgets.QLabel(self.frame_11)
        self.label_14.setGeometry(QtCore.QRect(200, 70, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.frame_11)
        self.label_15.setGeometry(QtCore.QRect(200, 90, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.frame_15 = QtWidgets.QFrame(self.frame_6)
        self.frame_15.setGeometry(QtCore.QRect(10, 90, 361, 81))
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.operator_alert_slider = QtWidgets.QSlider(self.frame_15)
        self.operator_alert_slider.setGeometry(QtCore.QRect(10, 50, 341, 22))
        self.operator_alert_slider.setMinimum(0)
        self.operator_alert_slider.setMaximum(100)
        self.operator_alert_slider.setSingleStep(25)
        self.operator_alert_slider.setPageStep(25)
        self.operator_alert_slider.setProperty("value", 50)
        self.operator_alert_slider.setOrientation(QtCore.Qt.Horizontal)
        self.operator_alert_slider.setInvertedAppearance(False)
        self.operator_alert_slider.setInvertedControls(False)
        self.operator_alert_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.operator_alert_slider.setTickInterval(25)
        self.operator_alert_slider.setObjectName("operator_alert_slider")
        self.outcome_threshold_value_4 = QtWidgets.QLabel(self.frame_15)
        self.outcome_threshold_value_4.setGeometry(QtCore.QRect(0, 20, 361, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.outcome_threshold_value_4.setFont(font)
        self.outcome_threshold_value_4.setAlignment(QtCore.Qt.AlignCenter)
        self.outcome_threshold_value_4.setObjectName("outcome_threshold_value_4")
        self.operator_alert_description = QtWidgets.QLabel(self.frame_15)
        self.operator_alert_description.setGeometry(QtCore.QRect(0, 0, 361, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.operator_alert_description.setFont(font)
        self.operator_alert_description.setAlignment(QtCore.Qt.AlignCenter)
        self.operator_alert_description.setObjectName("operator_alert_description")
        self.task_assessment_label = QtWidgets.QLabel(self.frame_6)
        self.task_assessment_label.setGeometry(QtCore.QRect(0, 0, 381, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.task_assessment_label.setFont(font)
        self.task_assessment_label.setAlignment(QtCore.Qt.AlignCenter)
        self.task_assessment_label.setObjectName("task_assessment_label")
        self.frame_10 = QtWidgets.QFrame(self.frame_6)
        self.frame_10.setGeometry(QtCore.QRect(10, 180, 361, 121))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_10.setObjectName("frame_10")
        self.label = QtWidgets.QLabel(self.frame_10)
        self.label.setGeometry(QtCore.QRect(10, 0, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.robot1_likelihood_label = QtWidgets.QLabel(self.frame_10)
        self.robot1_likelihood_label.setGeometry(QtCore.QRect(10, 30, 351, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.robot1_likelihood_label.setFont(font)
        self.robot1_likelihood_label.setObjectName("robot1_likelihood_label")
        self.label_4 = QtWidgets.QLabel(self.frame_10)
        self.label_4.setGeometry(QtCore.QRect(10, 50, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.robot1_crater_assessment_value = QtWidgets.QLabel(self.frame_10)
        self.robot1_crater_assessment_value.setGeometry(QtCore.QRect(80, 70, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.robot1_crater_assessment_value.setFont(font)
        self.robot1_crater_assessment_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.robot1_crater_assessment_value.setObjectName("robot1_crater_assessment_value")
        self.robot1_zone_assessment_value = QtWidgets.QLabel(self.frame_10)
        self.robot1_zone_assessment_value.setGeometry(QtCore.QRect(80, 90, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.robot1_zone_assessment_value.setFont(font)
        self.robot1_zone_assessment_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.robot1_zone_assessment_value.setObjectName("robot1_zone_assessment_value")
        self.label_5 = QtWidgets.QLabel(self.frame_10)
        self.label_5.setGeometry(QtCore.QRect(200, 70, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.frame_10)
        self.label_6.setGeometry(QtCore.QRect(200, 90, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.assessment_button = QtWidgets.QPushButton(self.frame_6)
        self.assessment_button.setGeometry(QtCore.QRect(90, 40, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.assessment_button.setFont(font)
        self.assessment_button.setObjectName("assessment_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1542, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.time_description_label.setText(_translate("MainWindow", "Elapsed Time:"))
        self.time_counter.setText(_translate("MainWindow", "0"))
        self.task_assessment_label_2.setText(_translate("MainWindow", "Mission Data"))
        self.task_assessment_label_3.setText(_translate("MainWindow", "Driving Control"))
        self.robot2_zones.setText(_translate("MainWindow", "-"))
        self.robot2_location.setText(_translate("MainWindow", "-"))
        self.robot2_state.setText(_translate("MainWindow", "-"))
        self.location_description_label_3.setText(_translate("MainWindow", "Location:"))
        self.robot2_craters.setText(_translate("MainWindow", "-"))
        self.state_description_label_3.setText(_translate("MainWindow", "Mode:"))
        self.hits_description_label_3.setText(_translate("MainWindow", "Craters hit:"))
        self.area_description_label_3.setText(_translate("MainWindow", "Zones hit:"))
        self.robot2_cargo_description_label.setText(_translate("MainWindow", "Cargo:"))
        self.robot2_cargo.setText(_translate("MainWindow", "-"))
        self.robot2_goArea2Button.setText(_translate("MainWindow", "Area 2"))
        self.robot2_goHomeButton.setText(_translate("MainWindow", "Home"))
        self.robot2_goArea1Button.setText(_translate("MainWindow", "Area 1"))
        self.robot2_goArea3Button.setText(_translate("MainWindow", "Area 3"))
        self.robot2_startControlButton.setText(_translate("MainWindow", "Start"))
        self.robot2_stopControlButton.setText(_translate("MainWindow", "Stop"))
        self.robot_zones.setText(_translate("MainWindow", "-"))
        self.robot_location.setText(_translate("MainWindow", "-"))
        self.robot_state.setText(_translate("MainWindow", "-"))
        self.location_description_label.setText(_translate("MainWindow", "Location:"))
        self.robot_craters.setText(_translate("MainWindow", "-"))
        self.state_description_label.setText(_translate("MainWindow", "Mode:"))
        self.hits_description_label.setText(_translate("MainWindow", "Craters hit:"))
        self.zones_description_label.setText(_translate("MainWindow", "Zones hit:"))
        self.cargo_description_label.setText(_translate("MainWindow", "Cargo:"))
        self.robot1_cargo.setText(_translate("MainWindow", "-"))
        self.goArea2Button.setText(_translate("MainWindow", "Area 2"))
        self.goHomeButton.setText(_translate("MainWindow", "Home"))
        self.goArea1Button.setText(_translate("MainWindow", "Area 1"))
        self.goArea3Button.setText(_translate("MainWindow", "Area 3"))
        self.startControlButton.setText(_translate("MainWindow", "Start"))
        self.stopControlButton.setText(_translate("MainWindow", "Stop"))
        self.delivery_description_label.setText(_translate("MainWindow", "Cargo Delivered:"))
        self.delivery_counter.setText(_translate("MainWindow", "0"))
        self.task_assessment_label_4.setText(_translate("MainWindow", "Color"))
        self.stopSimButton.setText(_translate("MainWindow", "Stop Simulation"))
        self.sendEventButton.setText(_translate("MainWindow", "Next Event"))
        self.dust_zone_color_label.setText(_translate("MainWindow", "Dust Zones"))
        self.crater_zone_color_label.setText(_translate("MainWindow", "Crater Zones"))
        self.blocked_zone_color_label.setText(_translate("MainWindow", "Blocked Zones"))
        self.report_button.setText(_translate("MainWindow", "Send Report"))
        self.report_slider_counter.setText(_translate("MainWindow", "-"))
        self.report_slider_value.setText(_translate("MainWindow", "-"))
        self.label_11.setText(_translate("MainWindow", "Report:"))
        self.robot2_likelihood_label.setText(_translate("MainWindow", "Likelihood of successful navigation to - : - "))
        self.label_13.setText(_translate("MainWindow", "Because the rover will encounter:"))
        self.robot2_crater_assessment_value.setText(_translate("MainWindow", "-"))
        self.robot2_zone_assessment_value.setText(_translate("MainWindow", "-"))
        self.label_14.setText(_translate("MainWindow", "crater zones"))
        self.label_15.setText(_translate("MainWindow", "dust zones"))
        self.outcome_threshold_value_4.setText(_translate("MainWindow", "-"))
        self.operator_alert_description.setText(_translate("MainWindow", "Alert when rove confidence is below:"))
        self.task_assessment_label.setText(_translate("MainWindow", "Robot Task Assessment"))
        self.label.setText(_translate("MainWindow", "Report:"))
        self.robot1_likelihood_label.setText(_translate("MainWindow", "Likelihood of successful navigation to - : - "))
        self.label_4.setText(_translate("MainWindow", "Because the rover will encounter:"))
        self.robot1_crater_assessment_value.setText(_translate("MainWindow", "-"))
        self.robot1_zone_assessment_value.setText(_translate("MainWindow", "-"))
        self.label_5.setText(_translate("MainWindow", "crater zones"))
        self.label_6.setText(_translate("MainWindow", "dust zones"))
        self.assessment_button.setText(_translate("MainWindow", "Run Assessment"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
