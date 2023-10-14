from PyQt5 import QtCore, QtGui, QtWidgets
import wmi
import sys
from run import Test, TestOne


def info_pc():
    computer = wmi.WMI()
    os_info = computer.Win32_OperatingSystem()[0]
    proc_info = computer.Win32_Processor()[0]
    gpu_info = computer.Win32_VideoController()[0]

    os_version = ' '.join([os_info.Version, os_info.BuildNumber])
    system_ram = float(os_info.TotalVisibleMemorySize) / 1048576  # KB to GB

    print('OS Version: {0}'.format(os_version))
    print('CPU: {0}'.format(proc_info.Name))
    print('GPU: {0}'.format(gpu_info.Name))
    print('RAM: {0:.0f} GB'.format(system_ram))


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(565, 270)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 60, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 100, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 140, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 180, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(30, 220, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")

        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(90, 60, 73, 22))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItems([""] * 9)

        self.comboBox_4 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_4.setGeometry(QtCore.QRect(90, 140, 73, 22))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.comboBox_4.setFont(font)
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItems([""] * 11)

        self.comboBox_6 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_6.setGeometry(QtCore.QRect(90, 220, 73, 22))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.comboBox_6.setFont(font)
        self.comboBox_6.setObjectName("comboBox_6")
        self.comboBox_6.addItems([""] * 11)

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(90, 20, 71, 22))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("1.0")

        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(90, 100, 71, 22))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setText("20.03")

        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(90, 180, 71, 22))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setText("0.12")

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(190, 20, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")

        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(190, 60, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")

        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(190, 100, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")

        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(190, 140, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")

        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(190, 180, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(340, 80, 93, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.btn_click_manual_test)

        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(340, 130, 93, 51))
        self.pushButton2.setObjectName("pushButton")
        self.pushButton2.clicked.connect(self.btn_click_test)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 565, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def btn_click_manual_test(self):
        f0 = float(self.lineEdit.text())
        tau = float(self.comboBox_2.currentText())
        t_imp = float(self.lineEdit_2.text())
        fs = float(self.comboBox_4.currentText())
        dts0 = float(self.comboBox_6.currentText())
        dt0 = float(self.lineEdit_3.text())
        test = TestOne()
        print(test)
        test.run_test(t_imp, tau, f0, dts0, dt0, fs)

    def btn_click_test(self):
        test = Test()
        test.testing()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "f0"))
        self.label_2.setText(_translate("MainWindow", "tau"))
        self.label_3.setText(_translate("MainWindow", "Tимп"))
        self.label_4.setText(_translate("MainWindow", "fs"))
        self.label_5.setText(_translate("MainWindow", "dT0"))
        self.label_6.setText(_translate("MainWindow", "dts0"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "1.0"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "1.1"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "1.2"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "1.3"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "1.4"))
        self.comboBox_2.setItemText(5, _translate("MainWindow", "1.5"))
        self.comboBox_2.setItemText(6, _translate("MainWindow", "1.6"))
        self.comboBox_2.setItemText(7, _translate("MainWindow", "1.7"))
        self.comboBox_2.setItemText(8, _translate("MainWindow", "1.8"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "8"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "10"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "20"))
        self.comboBox_4.setItemText(3, _translate("MainWindow", "40"))
        self.comboBox_4.setItemText(4, _translate("MainWindow", "60"))
        self.comboBox_4.setItemText(5, _translate("MainWindow", "80"))
        self.comboBox_4.setItemText(6, _translate("MainWindow", "100"))
        self.comboBox_4.setItemText(7, _translate("MainWindow", "200"))
        self.comboBox_4.setItemText(8, _translate("MainWindow", "250"))
        self.comboBox_4.setItemText(9, _translate("MainWindow", "500"))
        self.comboBox_4.setItemText(10, _translate("MainWindow", "1000"))
        self.comboBox_6.setItemText(0, _translate("MainWindow", "0.0"))
        self.comboBox_6.setItemText(1, _translate("MainWindow", "0.1"))
        self.comboBox_6.setItemText(2, _translate("MainWindow", "0.2"))
        self.comboBox_6.setItemText(3, _translate("MainWindow", "0.3"))
        self.comboBox_6.setItemText(4, _translate("MainWindow", "0.4"))
        self.comboBox_6.setItemText(5, _translate("MainWindow", "0.5"))
        self.comboBox_6.setItemText(6, _translate("MainWindow", "0.6"))
        self.comboBox_6.setItemText(7, _translate("MainWindow", "0.7"))
        self.comboBox_6.setItemText(8, _translate("MainWindow", "0.8"))
        self.comboBox_6.setItemText(9, _translate("MainWindow", "0.9"))
        self.comboBox_6.setItemText(10, _translate("MainWindow", "1.0"))
        self.label_7.setText(_translate("MainWindow", "МГц"))
        self.label_8.setText(_translate("MainWindow", "мкс"))
        self.label_9.setText(_translate("MainWindow", "мкс"))
        self.label_10.setText(_translate("MainWindow", "МГц"))
        self.label_11.setText(_translate("MainWindow", "мкс"))
        self.pushButton.setText(_translate("MainWindow", "RUN"))
        self.pushButton2.setText(_translate("MainWindow", "test"))


if __name__ == "__main__":
    info_pc()

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
