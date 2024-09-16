from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,QComboBox,QPushButton,QRadioButton,QButtonGroup, QWidget,QSlider,QAction,QFileDialog,QTableView,QLineEdit,QLabel,QRadioButton,QMessageBox
from PyQt5.QtGui import QStandardItemModel, QStandardItem,QQuaternion,QVector3D
from PyQt5.QtCore import Qt

import pyqtgraph.opengl as gl
import numpy as np
import sys
import open3d as o3d
from PIL import Image, ImageTk  # 画像データ用
import cv2
from sklearn.neighbors import NearestNeighbors

class PointCloudViewer(QMainWindow):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.scene().sigMouseClicked.connect(self.on_mouse_click)
        # メインウィンドウの設定
        self.setWindowTitle("Point Cloud Viewer")
        self.setGeometry(100, 100, 1200, 600)

        # メインウィジェット
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # メニューバーを作成
        menubar = self.menuBar()

        # ファイルメニューを作成
        file_menu = menubar.addMenu('File')

        # メニューアクションを追加
        imopen_action = QAction('Image data open', self)
        imopen_action.triggered.connect(self.menu_imagedataopen_clicked)
        imclear_action = QAction('Image data clear', self)
        imclear_action.triggered.connect(self.menu_imagedataclear_clicked)
        mtxopen_action = QAction('Mtx data open', self)
        mtxopen_action.triggered.connect(self.menu_mtxdataopen_clicked)
        rtopen_action = QAction('RT data open', self)
        rtopen_action.triggered.connect(self.menu_rtdataopen_clicked)
        open_action = QAction('3D data open', self)
        open_action.triggered.connect(self.menu_3ddataopen_clicked)
        clear_action = QAction('3D data clear', self)
        clear_action.triggered.connect(self.menu_3ddataclear_clicked)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)  # クリック時にウィンドウを閉じる


        file_menu.addAction(imopen_action)
        file_menu.addAction(imclear_action)
        file_menu.addAction(mtxopen_action)
        file_menu.addAction(rtopen_action)
        file_menu.addAction(open_action)
        file_menu.addAction(clear_action)
        file_menu.addAction(exit_action)

        # OpenGLビューウィジェットの作成
        # self.gl_widget = gl.GLViewWidget()
        self.gl_widget = ClickablePlotWidget()
        self.gl_widget.opts['rotationMethod'] = 'quaternion'

        eu = self.gl_widget.opts['rotation'].toEulerAngles()
        if self.gl_widget.opts['azimuth'] is not None:
            eu.setZ(-self.gl_widget.opts['azimuth']-90)
        if self.gl_widget.opts['elevation'] is not None:
            eu.setX(self.gl_widget.opts['elevation']-90)
        self.gl_widget.opts['rotation'] = QQuaternion.fromEulerAngles(eu)





        self.gl_widget.task1 = self.mousePressEvent_3dplot
        self.gl_widget.task2 = self.mouseReleaseEvent_3dplot
        self.gl_widget.task3 = self.mouseMoveEvent_3dplot
        self.gl_widget.task4 = self.mouseWheelEvent_3dplot
        self.gl_widget.task5 = self.keyPressEvent_3dplot
        self.gl_widget.task6 = self.keyReleaseEvent_3dplot

        self.Gridxy = gl.GLGridItem()
        self.Axes = gl.GLAxisItem()
        self.Axes.setSize(1,1,1)
        self.Gridxy.setSize(100,100,0.1)
        self.Gridxy.setSpacing(10,10,0)
        self.Gridxy.translate(0, 0, 0)

        self.Poss = []
        self.Cols = []
        self.GlobalInds = []

        self.gl_widget.addItem(self.Gridxy)
        self.gl_widget.addItem(self.Axes)

        self._downpos = []

        #ラジオボタン
        self.radio_button_group = QButtonGroup(self)
        self.radio_button_cam = QRadioButton('Camera View')
        self.radio_button_pos = QRadioButton('Points View')
        self.radio_button_cam.setChecked(True)
        self.radio_button_group.addButton(self.radio_button_cam)
        self.radio_button_group.addButton(self.radio_button_pos)
        self.radio_button_pos.toggled.connect(self.radio_change_view)
        self.radio_button_cam.toggled.connect(self.radio_change_view)
        # ボタンの作成
        self.button = QPushButton("Position Reset")
        self.button.setMaximumWidth(120)
        self.button.clicked.connect(self.on_button_click)
        # ボタンの作成
        self.button_seldel = QPushButton("SelectPoints Delete")
        self.button_seldel.setMaximumWidth(120)
        self.button_seldel.clicked.connect(self.on_button_seldel_click)

        # ボタンの作成
        self.button_setbase = QPushButton("◎")
        self.button_setbase.setMaximumWidth(30)
        self.button_setbase.clicked.connect(self.on_button_setbase_click)
        # ボタンの作成
        self.button_setori = QPushButton("Ori")
        self.button_setori.setMaximumWidth(30)
        self.button_setori.clicked.connect(self.on_button_setori_click)
        # ボタンの作成
        self.button_setp3p = QPushButton("P3P")
        self.button_setp3p.setMaximumWidth(35)
        self.button_setp3p.clicked.connect(self.on_button_setp3p_click)

        self.comboBox_p3p = QComboBox(self)
        self.comboBox_p3p.setMaximumWidth(35)
        self.comboBox_p3p.addItems([str(i) for i in range(1, 5)])  # 1から4までの数値を追加


        # ボタンの作成
        self.button_xu = QPushButton("+x")
        self.button_xu.setMaximumWidth(30)
        self.button_xu.clicked.connect(self.on_button_xu_click)
        # ボタンの作成
        self.button_xl = QPushButton("-x")
        self.button_xl.setMaximumWidth(30)
        self.button_xl.clicked.connect(self.on_button_xl_click)
        self.textbox_x = QLineEdit(self)
        self.textbox_x.setMaximumWidth(30)
        self.textbox_x.setInputMask("0.00")
        self.textbox_x.setText("0.1")
        # ボタンの作成
        self.button_yu = QPushButton("+y")
        self.button_yu.setMaximumWidth(30)
        self.button_yu.clicked.connect(self.on_button_yu_click)
        # ボタンの作成
        self.button_yl = QPushButton("-y")
        self.button_yl.setMaximumWidth(30)
        self.button_yl.clicked.connect(self.on_button_yl_click)
        self.textbox_y = QLineEdit(self)
        self.textbox_y.setMaximumWidth(30)
        self.textbox_y.setInputMask("0.00")
        self.textbox_y.setText("0.1")
        # ボタンの作成
        self.button_zu = QPushButton("+z")
        self.button_zu.setMaximumWidth(30)
        self.button_zu.clicked.connect(self.on_button_zu_click)
        # ボタンの作成
        self.button_zl = QPushButton("-z")
        self.button_zl.setMaximumWidth(30)
        self.button_zl.clicked.connect(self.on_button_zl_click)
        self.textbox_z = QLineEdit(self)
        self.textbox_z.setMaximumWidth(30)
        self.textbox_z.setInputMask("0.00")
        self.textbox_z.setText("0.1")
        # ボタンの作成
        self.button_eu = QPushButton("+R")
        self.button_eu.setMaximumWidth(30)
        self.button_eu.clicked.connect(self.on_button_eu_click)
        # ボタンの作成
        self.button_el = QPushButton("-R")
        self.button_el.setMaximumWidth(30)
        self.button_el.clicked.connect(self.on_button_el_click)
        self.textbox_e = QLineEdit(self)
        self.textbox_e.setMaximumWidth(30)
        self.textbox_e.setInputMask("0.00")
        self.textbox_e.setText("1.0")
        # ボタンの作成
        self.button_ru = QPushButton("+P")
        self.button_ru.setMaximumWidth(30)
        self.button_ru.clicked.connect(self.on_button_ru_click)
        # ボタンの作成
        self.button_rl = QPushButton("-P")
        self.button_rl.setMaximumWidth(30)
        self.button_rl.clicked.connect(self.on_button_rl_click)
        self.textbox_r = QLineEdit(self)
        self.textbox_r.setMaximumWidth(30)
        self.textbox_r.setInputMask("0.00")
        self.textbox_r.setText("1.0")
        # ボタンの作成
        self.button_au = QPushButton("+Y")
        self.button_au.setMaximumWidth(30)
        self.button_au.clicked.connect(self.on_button_au_click)
        # ボタンの作成
        self.button_al = QPushButton("-Y")
        self.button_al.setMaximumWidth(30)
        self.button_al.clicked.connect(self.on_button_al_click)
        self.textbox_a = QLineEdit(self)
        self.textbox_a.setMaximumWidth(30)
        self.textbox_a.setInputMask("0.00")
        self.textbox_a.setText("1.0")

        # ボタンの作成
        self.button_fovu = QPushButton("+fov")
        self.button_fovu.setMaximumWidth(30)
        self.button_fovu.clicked.connect(self.on_button_fovu_click)
        # ボタンの作成
        self.button_fovl = QPushButton("-fov")
        self.button_fovl.setMaximumWidth(30)
        self.button_fovl.clicked.connect(self.on_button_fovl_click)
        self.textbox_fov = QLineEdit(self)
        self.textbox_fov.setMaximumWidth(30)
        self.textbox_fov.setInputMask("00.00")
        self.textbox_fov.setText("1.0")

        # self.slider_3dpointsize = QSlider(Qt.Orientation.Vertical)
        self.slider_3dpointsize = QSlider(Qt.Orientation.Horizontal)
        self.slider_3dpointsize.setFixedSize(60,10)
        self.slider_3dpointsize.setMaximum(10)
        self.slider_3dpointsize.setMinimum(1)
        self.slider_3dpointsize.setValue(5)
        self.slider_3dpointsize.valueChanged.connect(self.slider_3dpointsize_valuechanged)
        self.label_3dpointsize = QLabel()
        self.label_3dpointsize.setFixedSize(80,30)
        self.label_3dpointsize.setText("PointSize:")

        self.slider_3dpointalp = QSlider(Qt.Orientation.Horizontal)
        self.slider_3dpointalp.setFixedSize(60,10)
        self.slider_3dpointalp.setMaximum(100)
        self.slider_3dpointalp.setMinimum(0)
        self.slider_3dpointalp.setValue(100)
        self.slider_3dpointalp.valueChanged.connect(self.slider_3dpointalp_valuechanged)
        self.label_3dpointalp = QLabel()
        self.label_3dpointalp.setFixedSize(80,30)
        self.label_3dpointalp.setText("Alpha:")

        self.slider_3dpointfov = QSlider(Qt.Orientation.Horizontal)
        self.slider_3dpointfov.setFixedSize(60,10)
        self.slider_3dpointfov.setMaximum(17000)
        self.slider_3dpointfov.setMinimum(3000)
        self.slider_3dpointfov.setValue(6000)
        self.slider_3dpointfov.valueChanged.connect(self.slider_3dpointfov_valuechanged)
        self.label_3dpointfov = QLabel()
        self.label_3dpointfov.setFixedSize(80,30)
        self.label_3dpointfov.setText("FOV:")

        self.label_position = QLabel()
        self.label_position.setFixedSize(200,30)
        self.label_position.setText("Position:")

        self.label_rotation = QLabel()
        self.label_rotation.setFixedSize(200,30)
        self.label_rotation.setText("Rotation:")
        #Viewを作成
        # QTableViewとそのモデルを作成
        self.tableView_main = QTableView(self)
        self.model = QStandardItemModel(self)

        # カラムを設定
        self.model.setColumnCount(8)
        self.model.setHorizontalHeaderLabels(["▽", "X", "Y", "Z", "index","px" ,"py" , "メモ"])

        self.tableView_main.setFixedSize(330, 300)

        self.tableView_main.setModel(self.model)
        for column in range(self.model.columnCount()):
            self.tableView_main.setColumnWidth(column, 40)

        self.tableView_main.selectionModel().selectionChanged.connect(self.table_selected_row)

        self.gl_image = None
        self.Plot = None
        self.Plotr = None
        self.Poss_np = None
        self.selpos_index = None
        self.Base2Poss_rot = None
        self.Base2Poss_tra = None
        self.cv_image = None
        self.key_ctrl = 0
        

        self.mtx_data = None
        # レイアウトの作成
        layout = QHBoxLayout(self.main_widget)

        # layout_0 = QHBoxLayout(self.main_widget)
        layout_1 = QVBoxLayout()
        layout_2 = QHBoxLayout()
        layout_3 = QHBoxLayout()
        layout_4 = QHBoxLayout()
        layout_5 = QHBoxLayout()
        layout_6 = QHBoxLayout()
        layout_7 = QHBoxLayout()
        layout_8 = QHBoxLayout()
        layout_9 = QHBoxLayout()
        layout_10 = QHBoxLayout()
        layout_11 = QHBoxLayout()
        layout_12 = QHBoxLayout()

        layout_1.addWidget(self.button)
        layout_1.addWidget(self.radio_button_cam)
        layout_1.addWidget(self.radio_button_pos)
        layout_2.addWidget(self.button_xu)
        layout_2.addWidget(self.textbox_x)
        layout_2.addWidget(self.button_xl)
        layout_3.addWidget(self.button_yu)
        layout_3.addWidget(self.textbox_y)
        layout_3.addWidget(self.button_yl)
        layout_4.addWidget(self.button_zu)
        layout_4.addWidget(self.textbox_z)
        layout_4.addWidget(self.button_zl)
        layout_5.addWidget(self.button_eu)
        layout_5.addWidget(self.textbox_e)
        layout_5.addWidget(self.button_el)
        layout_6.addWidget(self.button_ru)
        layout_6.addWidget(self.textbox_r)
        layout_6.addWidget(self.button_rl)
        layout_7.addWidget(self.button_au)
        layout_7.addWidget(self.textbox_a)
        layout_7.addWidget(self.button_al)

        layout_8.addWidget(self.button_fovu)
        layout_8.addWidget(self.textbox_fov)
        layout_8.addWidget(self.button_fovl)
        layout_9.addWidget(self.label_3dpointfov)
        layout_9.addWidget(self.slider_3dpointfov)

        layout_10.addWidget(self.label_3dpointsize)
        layout_10.addWidget(self.slider_3dpointsize)

        layout_11.addWidget(self.label_3dpointalp)
        layout_11.addWidget(self.slider_3dpointalp)

        layout_1.addLayout(layout_2)
        layout_1.addLayout(layout_3)
        layout_1.addLayout(layout_4)
        layout_1.addLayout(layout_5)
        layout_1.addLayout(layout_6)
        layout_1.addLayout(layout_7)
        layout_1.addLayout(layout_8)
        layout_1.addLayout(layout_9)
        layout_1.addLayout(layout_10)
        layout_1.addLayout(layout_11)
        layout_1.addWidget(self.label_position)
        layout_1.addWidget(self.label_rotation)

        # layout_1.addWidget(self.slider_3dpointsize)
        # layout_1.addWidget(self.slider_3dpointalp)
        layout_1.addWidget(self.button_seldel)
        layout_12.addWidget(self.button_setbase)
        layout_12.addWidget(self.button_setori)
        layout_12.addWidget(self.button_setp3p)
        layout_12.addWidget(self.comboBox_p3p)
        layout_1.addLayout(layout_12)

        layout_1.addWidget(self.tableView_main)


        layout.addLayout(layout_1)
        layout.addWidget(self.gl_widget)

        # self.gl_widget.opts['fov'] = 120

        self.setWindowTitle('Pick 3D points')
        self.show()
    def menu_3ddataopen_clicked(self, event=None):
        data3d_filename = QFileDialog.getOpenFileName(None,"ファイルを選択してね","","3D point Files(*.ply)")
        if data3d_filename[0] == "":
            return

        if self.Plot is None:
            self.plotGLPlot(data3d_filename[0])
        else:
            for item in self.gl_widget.items:
                if self.Plot == item:
                    self.gl_widget.removeItem(self.Plot)
            self.plotGLPlot(data3d_filename[0])

    #Viewをクリックしたときの行の位置を取得
    def tableviewClicked(self, indexClicked):
        self.selectRow = indexClicked.row()

    def menu_3ddataclear_clicked(self, event=None):
        if self.Plot is None:
            return
        for item in self.gl_widget.items:
            if self.Plot == item:
                self.gl_widget.removeItem(self.Plot)

    def menu_imagedataopen_clicked(self, event=None):
        dataim_filename = QFileDialog.getOpenFileName(None,"ファイルを選択してね","","Image Files(*.jpg);;Image Files(*.JPG);;Image Files(*.bmp);;Image Files(*.BMP)")
        if dataim_filename[0] == "":
            return
        if self.cv_image is None:
            self.plotImPlot(dataim_filename[0])
        else:
            for item in self.gl_widget.items:
                if self.overlay_image == item:
                    self.gl_widget.removeItem(self.overlay_image)
            self.plotImPlot(dataim_filename[0])

    def menu_imagedataclear_clicked(self, event=None):
        if self.cv_image is None:
            return
        for item in self.gl_widget.items:
            if self.overlay_image == item:
                self.gl_widget.removeItem(self.overlay_image)
    def menu_mtxdataopen_clicked(self):
        dataim_filename = QFileDialog.getOpenFileName(None,"ファイルを選択してね","","CSV Files(*.csv);;CSV Files(*.CSV)")
        if dataim_filename[0] == "":
            return
        try:
            r = np.loadtxt(dataim_filename[0], delimiter=',')
        except Exception as e:
            raise e
        self.mtx_data = r

        if self.gl_image is None:
            return
        self.cv_image = np.array(self.pil_image)
        height,width = self.cv_image.shape[:2]
        tx = width/2-self.mtx_data[0,2]
        ty = height/2-self.mtx_data[1,2]
        mv_mat = np.float32([[1,0,tx],[0,1,ty]])
        self.cv_image = cv2.warpAffine(self.cv_image,mv_mat,(width,height))
        self.gl_image = self.cv_image.astype(np.uint8)
        self.gl_image = np.pad(self.gl_image, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=255)
        self.overlay_image.setData(self.gl_image)
        # self.cv_image = np.array(self.pil_image)

        self.gl_widget.opts['fov'] = 2.0*np.arctan2(width,2.0*self.mtx_data[0,0])/np.pi*180.0
        self.slider_3dpointfov.setValue(int(self.gl_widget.opts['fov']*100.0))
        text_buf = "FOV: %3.2f"%float(self.gl_widget.opts['fov'])
        self.label_3dpointfov.setText(text_buf)
        self.set_imagebackground()
    def menu_rtdataopen_clicked(self):
        dataim_filename = QFileDialog.getOpenFileName(None,"ファイルを選択してね","","CSV Files(*.csv);;CSV Files(*.CSV)")
        if dataim_filename[0] == "":
            return
        # filename = tk.filedialog.askopenfilename(
        #             filetypes = [("CSV file", ".csv")], # ファイルフィルタ
        #             initialdir = os.getcwd() # カレントディレクトリ
        #             )
        # if filename == "":
        #     return
        # try:
        #     r = np.loadtxt(filename, delimiter=',')
        # except Exception as e:
        #     raise e
        # self.mtx_data = r

    def on_button_click(self):
        if self.Poss_np is None:
            return
        self.reset_rottra()
        self.update_pos()

    def on_button_xu_click(self):
        if self.radio_button_cam.isChecked():
            self.gl_widget.pan(float(self.textbox_x.text()),0,0)
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_trax(float(self.textbox_x.text()))
                self.update_pos()

    def on_button_xl_click(self):
        if self.radio_button_cam.isChecked():
            self.gl_widget.pan(-float(self.textbox_x.text()),0,0)
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_trax(-float(self.textbox_x.text()))
                self.update_pos()

    def on_button_yu_click(self):
        if self.radio_button_cam.isChecked():
            self.gl_widget.pan(0,float(self.textbox_y.text()),0)
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_tray(float(self.textbox_y.text()))
                self.update_pos()

    def on_button_yl_click(self):
        if self.radio_button_cam.isChecked():
            self.gl_widget.pan(0,-float(self.textbox_y.text()),0)
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_tray(-float(self.textbox_y.text()))
                self.update_pos()

    def on_button_zu_click(self):
        if self.radio_button_cam.isChecked():
            self.gl_widget.pan(0,0,float(self.textbox_z.text()))
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_traz(float(self.textbox_z.text()))
                self.update_pos()

    def on_button_zl_click(self):
        if self.radio_button_cam.isChecked():
            self.gl_widget.pan(0,0,-float(self.textbox_z.text()))
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_traz(-float(self.textbox_z.text()))
                self.update_pos()
    def on_button_eu_click(self):
        if self.radio_button_cam.isChecked():
            q = QQuaternion.fromEulerAngles(float(self.textbox_e.text()),0,0)
            q *= self.gl_widget.opts['rotation']
            self.gl_widget.opts['rotation'] = q

            dist = np.array([0,0,self.gl_widget.opts['distance']])
            rot_q = QQuaternion.toRotationMatrix(self.gl_widget.opts['rotation'])
            rot_np = np.zeros([3,3])
            rot_np[0,0] = rot_q[0,0]
            rot_np[0,1] = rot_q[0,1]
            rot_np[0,2] = rot_q[0,2]
            rot_np[1,0] = rot_q[1,0]
            rot_np[1,1] = rot_q[1,1]
            rot_np[1,2] = rot_q[1,2]
            rot_np[2,0] = rot_q[2,0]
            rot_np[2,1] = rot_q[2,1]
            rot_np[2,2] = rot_q[2,2]
            C = self.gl_widget.opts['center']
            c0 = np.array([C[0],C[1],C[2]])
            campos = c0+dist@rot_np



            self.gl_widget.update()
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_rotx(float(self.textbox_e.text())/180.0*np.pi)
                self.update_pos()
    def on_button_el_click(self):
        if self.radio_button_cam.isChecked():
            q = QQuaternion.fromEulerAngles(-float(self.textbox_e.text()),0,0)
            q *= self.gl_widget.opts['rotation']
            self.gl_widget.opts['rotation'] = q
            self.gl_widget.update()
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_rotx(-float(self.textbox_e.text())/180.0*np.pi)
                self.update_pos()
    def on_button_ru_click(self):
        if self.radio_button_cam.isChecked():
            q = QQuaternion.fromEulerAngles(0,float(self.textbox_r.text()),0)
            q *= self.gl_widget.opts['rotation']
            self.gl_widget.opts['rotation'] = q
            self.gl_widget.update()
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_roty(float(self.textbox_r.text())/180.0*np.pi)
                self.update_pos()
    def on_button_rl_click(self):
        if self.radio_button_cam.isChecked():
            q = QQuaternion.fromEulerAngles(0,-float(self.textbox_r.text()),0)
            q *= self.gl_widget.opts['rotation']
            self.gl_widget.opts['rotation'] = q
            self.gl_widget.update()
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_roty(-float(self.textbox_r.text())/180.0*np.pi)
                self.update_pos()
    def on_button_au_click(self):
        if self.radio_button_cam.isChecked():
            q = QQuaternion.fromEulerAngles(0,0,float(self.textbox_a.text()))
            q *= self.gl_widget.opts['rotation']
            self.gl_widget.opts['rotation'] = q
            self.gl_widget.update()
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_rotz(float(self.textbox_a.text())/180.0*np.pi)
                self.update_pos()
    def on_button_al_click(self):
        if self.radio_button_cam.isChecked():
            q = QQuaternion.fromEulerAngles(0,0,-float(self.textbox_a.text()))
            q *= self.gl_widget.opts['rotation']
            self.gl_widget.opts['rotation'] = q
            self.gl_widget.update()
            self.set_imagebackground()
        if self.Poss_np is not None:
            if self.radio_button_pos.isChecked():
                self.scatter_rotz(-float(self.textbox_a.text())/180.0*np.pi)
                self.update_pos()

    def radio_change_view(self):
        # どのラジオボタンが選択されているかに基づいてカメラビューを変更
        if self.radio_button_cam.isChecked():
            print("s")
        elif self.radio_button_pos.isChecked():
            print("a")

    def slider_3dpointsize_valuechanged(self):
        if self.Plot is None:
            return
        self.Plot.setData(size = self.slider_3dpointsize.value()/100.0)
        text_buf = "PointSize: %3.2f"%float(self.slider_3dpointsize.value()/100.0)
        self.label_3dpointsize.setText(text_buf)

    def slider_3dpointalp_valuechanged(self):
        if self.Plot is None:
            return
        color_data_l = np.append(self.color_data,self.alpha_data*self.slider_3dpointalp.value()/100.0,axis=1)
        self.Plot.setData(color = color_data_l)
        text_buf = "Alpha: %3.2f"%float(self.slider_3dpointalp.value()/100.0)
        self.label_3dpointalp.setText(text_buf)

    def on_button_fovu_click(self):
        fov_diff = float(self.textbox_fov.text())
        self.slider_3dpointfov.setValue(int(100.0*(self.slider_3dpointfov.value()/100.0 + fov_diff)))
        self.gl_widget.opts['fov'] = self.slider_3dpointfov.value()/100.0
        self.set_imagebackground()
        self.gl_widget.update()
    def on_button_fovl_click(self):
        fov_diff = -float(self.textbox_fov.text())
        self.slider_3dpointfov.setValue(int(100.0*(self.slider_3dpointfov.value()/100.0 + fov_diff)))
        self.gl_widget.opts['fov'] = self.slider_3dpointfov.value()/100.0
        self.set_imagebackground()
        self.gl_widget.update()

    def slider_3dpointfov_valuechanged(self):
        self.gl_widget.opts['fov'] = self.slider_3dpointfov.value()/100.0
        self.set_imagebackground()
        self.gl_widget.update()
        text_buf = "FOV: %3.2f"%float(self.slider_3dpointfov.value()/100.0)
        self.label_3dpointfov.setText(text_buf)

    def mousePressEvent_3dplot(self,event):
        self._downpos = event.pos()
    def mouseReleaseEvent_3dplot(self,event):
        if self._downpos == event.pos():
            x = event.pos().x()
            y = event.pos().y()
            if event.button() == 2 :

                self.mPosition()
                self.get_imgpoint()

            elif event.button() == 1:
                x = x - self.width() / 2.0
                y = y - self.height() / 2.0

        self._prev_zoom_pos = None
        self._prev_pan_pos = None
        self.set_imagebackground()

    def mouseMoveEvent_3dplot(self,event):
        self.set_imagebackground()

    def mouseWheelEvent_3dplot(self,event):
        self.set_imagebackground()
    def keyPressEvent_3dplot(self,event):
        if event.key() == 16777249:
            self.key_ctrl = 1
    def keyReleaseEvent_3dplot(self,event):
        if event.key() == 16777249:
            self.key_ctrl = 0
    def keyPressEvent(self, event):
        if event.key() == 16777249:
            self.key_ctrl = 1
    def keyReleaseEvent(self, event):
        if event.key() == 16777249:
            self.key_ctrl = 0
    def rot_zyx(self,yaw,pitch,roll):
        rot_x = np.array([[ 1,                 0,                  0],
                    [ 0, np.cos(roll), -np.sin(roll)],
                    [ 0, np.sin(roll),  np.cos(roll)]])
        rot_y = np.array([[ np.cos(pitch), 0,  np.sin(pitch)],
                    [                 0, 1,                  0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
        rot_z = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                    [ np.sin(yaw),  np.cos(yaw), 0],
                    [                 0,                  0, 1]])
        return rot_z@rot_y@rot_x
    def rot_xyz(self,roll,pitch,yaw):
        rot_x = np.array([[ 1,                 0,                  0],
                    [ 0, np.cos(roll), -np.sin(roll)],
                    [ 0, np.sin(roll),  np.cos(roll)]])
        rot_y = np.array([[ np.cos(pitch), 0,  np.sin(pitch)],
                    [                 0, 1,                  0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
        rot_z = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                    [ np.sin(yaw),  np.cos(yaw), 0],
                    [                 0,                  0, 1]])
        return rot_x@rot_y@rot_z
    def q2rot(self,w,x,y,z):
        rot = np.zeros([3,3])
        rot[0,0] = 2*w**2+2*x**2-1
        rot[0,1] = 2*x*y-2*z*w
        rot[0,2] = 2*x*z+2*y*w

        rot[1,0] = 2*x*y+2*z*w
        rot[1,1] = 2*w**2+2*y**2-1
        rot[1,2] = 2*y*z-2*x*w

        rot[2,0] = 2*x*z-2*y*w
        rot[2,1] = 2*y*z+2*x*w
        rot[2,2] = 2*w**2+2*z**2-1
        return rot
    def set_imagebackground(self):
        if self.gl_image is not None:
            camera_dist = -self.gl_image.shape[1]/(2.0*np.tan(self.gl_widget.opts['fov']/180.0*np.pi/2.0)) + self.gl_widget.opts['distance']

            pos_buf = self.gl_widget.opts['center']
            self.overlay_image.resetTransform()
            self.overlay_image.translate(-self.gl_image.shape[0]*0.5, -self.gl_image.shape[1]*0.5,camera_dist, local=False)  # 画像の位置を調整

            if self.gl_widget.opts['rotationMethod'] == "quaternion":
                roll  = self.gl_widget.opts['rotation'].toEulerAngles()[0]
                pitch = self.gl_widget.opts['rotation'].toEulerAngles()[1]
                yaw   = self.gl_widget.opts['rotation'].toEulerAngles()[2]
                self.gl_widget.opts['elevation'] = roll - 90
                self.gl_widget.opts['azimuth'] = -yaw - 90

                self.overlay_image.rotate(pitch ,1, 0, 0, local=False)  # 画像の位置を調整
                self.overlay_image.rotate(-roll,0, 1, 0, local=False)  # 画像の位置を調整
                self.overlay_image.rotate(-yaw-90  ,0, 0, 1, local=False)  # 画像の位置を調整
                self.overlay_image.translate(pos_buf[0],pos_buf[1],pos_buf[2], local=False)  # 画像の位置を調整
            else:
                self.overlay_image.rotate(0,1, 0, 0, local=False)  # 画像の位置を調整
                self.overlay_image.rotate(-self.gl_widget.opts['elevation']+90,0, 1, 0, local=False)  # 画像の位置を調整
                self.overlay_image.rotate(self.gl_widget.opts['azimuth'],0, 0, 1, local=False)  # 画像の位置を調整
                self.overlay_image.translate(pos_buf[0],pos_buf[1],pos_buf[2], local=False)  # 画像の位置を調整
        if self.Plot is not None:
            if self.gl_widget.opts['rotationMethod'] == "quaternion":
                roll  = self.gl_widget.opts['rotation'].toEulerAngles()[0]/180.0*np.pi
                pitch = self.gl_widget.opts['rotation'].toEulerAngles()[1]/180.0*np.pi
                yaw   = self.gl_widget.opts['rotation'].toEulerAngles()[2]/180.0*np.pi

                rotcam = self.rot_xyz(roll,pitch,yaw)
                dist = np.array([0,0,self.gl_widget.opts['distance']])
                rot_q = QQuaternion.toRotationMatrix(self.gl_widget.opts['rotation'])
                rot_np = np.zeros([3,3])
                rot_np[0,0] = rot_q[0,0]
                rot_np[0,1] = rot_q[0,1]
                rot_np[0,2] = rot_q[0,2]
                rot_np[1,0] = rot_q[1,0]
                rot_np[1,1] = rot_q[1,1]
                rot_np[1,2] = rot_q[1,2]
                rot_np[2,0] = rot_q[2,0]
                rot_np[2,1] = rot_q[2,1]
                rot_np[2,2] = rot_q[2,2]
                C = self.gl_widget.opts['center']
                c0 = np.array([C[0],C[1],C[2]])
                campos = c0+dist@rot_np
                # campos = c0
            else:
                campos = np.array([self.gl_widget.cameraPosition()[0],self.gl_widget.cameraPosition()[1],self.gl_widget.cameraPosition()[2]])
                roll = self.gl_widget.opts["elevation"]/180.0*np.pi
                yaw = self.gl_widget.opts["azimuth"]/180.0*np.pi
                rotcam = self.rot_xyz(roll,0,yaw)
            cam2obj_pos = self.Base2Poss_tra@self.Base2Poss_rot - campos
            cam2obj_pos = cam2obj_pos @ rotcam.T
            # postxt_x = "{:.3f}".format(cam2obj_pos[0,0])
            # postxt_y = "{:.3f}".format(cam2obj_pos[0,1])
            # postxt_z = "{:.3f}".format(cam2obj_pos[0,2])
            postxt_x = "{:.3f}".format(campos[0])
            postxt_y = "{:.3f}".format(campos[1])
            postxt_z = "{:.3f}".format(campos[2])
            eu = QQuaternion.toEulerAngles(self.gl_widget.opts['rotation'])
            rottxt_x = "{:.3f}".format(eu[0])
            rottxt_y = "{:.3f}".format(eu[1])
            rottxt_z = "{:.3f}".format(eu[2])
            self.label_position.setText("Position:"+postxt_x+" "+postxt_y+" "+postxt_z)
            self.label_rotation.setText("Rotation:"+rottxt_x+" "+rottxt_y+" "+rottxt_z)

    def get_imgpoint(self):
        if self.key_ctrl == 1:
            if self.cv_image is None:
                return
            mx = self._downpos.x()
            my = self._downpos.y()
            view_w = self.gl_widget.width()
            view_h = self.gl_widget.height()

            px = int((2.0 * mx / view_w - 1.0) * self.cv_image.shape[1]/2.0+self.cv_image.shape[1]/2)
            py = int((2.0 * my / view_h - 1.0) * view_h / view_w * self.cv_image.shape[1]/2.0+self.cv_image.shape[0]/2)
            px_item = QStandardItem()
            py_item = QStandardItem()
            px_item.setTextAlignment(Qt.AlignCenter)
            px_item.setEditable(False)  # 編集不可に設定
            py_item.setTextAlignment(Qt.AlignCenter)
            py_item.setEditable(False)  # 編集不可に設定
            for row in range(self.model.rowCount()):
                item = self.model.item(row, 0)
                if item.checkState() == Qt.Checked:
                    px_item.setText(str(px))
                    py_item.setText(str(self.cv_image.shape[0] - py))
                    self.model.setItem(row,5,px_item)
                    self.model.setItem(row,6,py_item)
                    break

            cv_image_c = self.cv_image.copy()
            cv2.drawMarker(cv_image_c, np.array([int(px),int(py)]), (255, 0, 0))
            cv_image_c = cv_image_c.astype(np.uint8)
            cv_image_c = np.pad(cv_image_c, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=255)
            self.overlay_image.setData(cv_image_c)
    def plotImPlot(self,file_name):

        # PIL.Imageで開く
        self.pil_image = Image.open(file_name)
        # PillowからNumPy(OpenCVの画像)へ変換
        self.cv_image = np.array(self.pil_image)
        height,width = self.cv_image.shape[:2]
        if self.mtx_data is None:
            self.mtx_data = np.zeros([3,3])
            fx = width/(2.0*self.gl_widget.opts['fov']/180.0*np.pi)
            self.mtx_data[0,0] = fx
            self.mtx_data[1,1] = fx
            self.mtx_data[2,2] = 1
            self.mtx_data[0,2] = width/2
            self.mtx_data[1,2] = height/2
        else:
            self.gl_widget.opts['fov'] = 2.0*np.arctan2(width,2.0*self.mtx_data[0,0])/np.pi*180.0
            self.slider_3dpointfov.setValue(int(self.gl_widget.opts['fov']*100.0))
            text_buf = "FOV: %3.2f"%float(self.gl_widget.opts['fov'])
            self.label_3dpointfov.setText(text_buf)


        tx = width/2-self.mtx_data[0,2]
        ty = height/2-self.mtx_data[1,2]
        mv_mat = np.float32([[1,0,tx],[0,1,ty]])
        cv_image_c = cv2.warpAffine(self.cv_image,mv_mat,(width,height))

        self.gl_image = cv_image_c.astype(np.uint8)
        self.gl_image = np.pad(self.gl_image, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=255)

        self.overlay_image = gl.GLImageItem(self.gl_image)
        self.set_imagebackground()
        self.gl_widget.addItem(self.overlay_image)

    def plotGLPlot(self,file_name):
        ptCloud = o3d.io.read_point_cloud(file_name)
        self.color_data = np.asarray(ptCloud.colors)
        length = len(self.color_data)
        self.alpha_data = np.ones([length,1])
        color_data_l = np.append(self.color_data,self.alpha_data,axis=1)

        self.Poss_np = np.asarray(ptCloud.points)
        self.Poss_np[:,2] = -self.Poss_np[:,2]
        self.Base_Poss_np = np.asarray(ptCloud.points)
        self.Base_Poss_np[:,2] = -self.Base_Poss_np[:,2]
        self.Base2Poss_rot = np.eye(3)
        self.Base2Poss_tra = np.zeros([1,3])

        self.Poss_npmax = self.Poss_np.max(axis=0)
        self.Poss_npmin = self.Poss_np.min(axis=0)

        self.GlobalInds = list(range(0,length))
        # self.Poss = self.Poss_np.tolist()

        self.Plot = gl.GLScatterPlotItem()
        # self.Plot.setData(pos=poss, size=point_size, color=cols, pxMode=False)
        self.Plot.setData(pos=self.Poss_np, size=self.slider_3dpointsize.value()/100.0, color=color_data_l, pxMode=False)
        # self.Plot.setGLOptions('opaque')
        self.Plot.setGLOptions('translucent')
        # self.Plot.setGLOptions('additive')
        self.gl_widget.addItem(self.Plot)

    def mPosition(self):
        if self.key_ctrl == 0:
            if self.Poss_np is None:
                return
            mx = self._downpos.x()
            my = self._downpos.y()
            self.Candidates = []
            self.Dist = []

            view_w = self.gl_widget.width()
            view_h = self.gl_widget.height()

            x = 2.0 * mx / view_w - 1.0
            y = 1.0 - (2.0 * my / view_h)
            z = 1.0

            PM = np.matrix([self.gl_widget.projectionMatrix().data()[0:4],
                            self.gl_widget.projectionMatrix().data()[4:8],
                            self.gl_widget.projectionMatrix().data()[8:12],
                            self.gl_widget.projectionMatrix().data()[12:16]])

            PMi = np.linalg.inv(PM)
            VM = np.matrix([self.gl_widget.viewMatrix().data()[0:4],
                            self.gl_widget.viewMatrix().data()[4:8],
                            self.gl_widget.viewMatrix().data()[8:12],
                            self.gl_widget.viewMatrix().data()[12:16]])


            ray_clip = np.matrix([x, y, -1.0, 1.0]).T
            ray_eye = PMi * ray_clip
            ray_eye[2] = -1
            ray_eye[3] = 0
            ray_world = VM * ray_eye
            ray_world = ray_world[0:3].T
            ray_world = ray_world / np.linalg.norm(ray_world)
            # O = np.matrix(self.gl_widget.cameraPosition())
            dist = np.array([0,0,self.gl_widget.opts['distance']])
            rot_q = QQuaternion.toRotationMatrix(self.gl_widget.opts['rotation'])
            rot_np = np.zeros([3,3])
            rot_np[0,0] = rot_q[0,0]
            rot_np[0,1] = rot_q[0,1]
            rot_np[0,2] = rot_q[0,2]
            rot_np[1,0] = rot_q[1,0]
            rot_np[1,1] = rot_q[1,1]
            rot_np[1,2] = rot_q[1,2]
            rot_np[2,0] = rot_q[2,0]
            rot_np[2,1] = rot_q[2,1]
            rot_np[2,2] = rot_q[2,2]
            C = self.gl_widget.opts['center']
            c0 = np.array([C[0],C[1],C[2]])
            O = c0+dist@rot_np
            p0 = np.array([O[0],O[1],O[2]])  # 直線上の点
            d = np.array([ray_world[0,0],ray_world[0,1],ray_world[0,2]])  # 直線の方向ベクトル
            self.Poss_npmax = self.Poss_np.max(axis=0)
            self.Poss_npmin = self.Poss_np.min(axis=0)
            ###########################
            t_length = np.sqrt((self.Poss_npmax[0]-self.Poss_npmin[0])**2+(self.Poss_npmax[1]-self.Poss_npmin[1])**2+(self.Poss_npmax[2]-self.Poss_npmin[2])**2)
            t_means = np.array([(self.Poss_npmax[0]+self.Poss_npmin[0])/2.0,(self.Poss_npmax[1]+self.Poss_npmin[1])/2.0,(self.Poss_npmax[2]+self.Poss_npmin[2])/2.0])
            t_values = np.linspace(-t_length/2, t_length/2, 1000)
            line_points = np.array([p0 + (t + np.linalg.norm(p0 - t_means)) * d for t in t_values])
            # t_values = np.linspace(0, t_length, 1000)
            # line_points = np.array([p0 + t * d for t in t_values])
            # k近傍法を使用して直線に最も近い点を求める
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.Poss_np)
            distances, indices = nbrs.kneighbors(line_points)

            # 最近傍点の取得
            # d_esp = 0.01
            # min_index = 0
            # for i in range(int(len(t_values))):
            #     mid_p = i
            #     if distances[mid_p] < d_esp:
            #         min_index = mid_p
            #         break
            min_index = np.argmin(distances)
            self.add_row(indices.flatten()[min_index])

    def add_row(self, number=None):
        if number is None:
            number = self.model.rowCount()

        # チェックボックス
        check_item = QStandardItem()
        check_item.setCheckable(True)
        check_item.setCheckState(Qt.Unchecked)
        check_item.setEditable(False)  # 編集不可に設定

        # 数値
        number_item = []
        for i in range(6):
            number_item.append(QStandardItem())
            number_item[i].setTextAlignment(Qt.AlignCenter)
            number_item[i].setEditable(False)  # 編集不可に設定

        number_item[0].setText("{:.3f}".format(self.Base_Poss_np[number,0]))
        number_item[1].setText("{:.3f}".format(self.Base_Poss_np[number,1]))
        number_item[2].setText("{:.3f}".format(self.Base_Poss_np[number,2]))
        number_item[3].setText(str(number))
        number_item[4].setText("")
        number_item[5].setText("")
        # 文字列
        string_item = QStandardItem("")

        # テーブルにアイテムを追加
        self.model.appendRow([check_item, number_item[0], number_item[1], number_item[2], number_item[3], number_item[4], number_item[5], string_item])
        new_index = self.model.index(self.model.rowCount()-1, 0)  # 新しい行のインデックスを取得
        self.tableView_main.setCurrentIndex(new_index)
        self.tableView_main.selectRow(self.model.rowCount()-1)  # 行全体を選択状態にする

    def table_selected_row(self):
        selected_indexes = self.tableView_main.selectionModel().selectedRows()
        self.selpos_index = np.zeros(len(selected_indexes),dtype=int)
        i = 0
        for index in selected_indexes:
            number_value = int(self.model.item(index.row(),4).text())
            self.selpos_index[i] = number_value
            i = i + 1
        self.update_selpos()
    def on_button_setbase_click(self):
        self.Base_Poss_np = self.Poss_np
        self.reset_rottra()
        self.update_pos()

    def on_button_setori_click(self):
        rows_to_sel = []
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            if item.checkState() == Qt.Checked:
                rows_to_sel.append(row)
        if len(rows_to_sel) == 0:
            return
        cpos = np.zeros([len(rows_to_sel),3])
        i = 0
        for row in reversed(rows_to_sel):
            cpos[i,:] = self.Poss_np[int(self.model.item(row,4).text()),:]
            i = i + 1
        if self.Base2Poss_tra is None or self.Base2Poss_rot is None:
            return
        if self.Poss_np is None:
            return
        self.Base2Poss_tra = self.Base2Poss_tra - np.mean(cpos,axis = 0)@self.Base2Poss_rot
        self.update_pos()

    def on_button_setp3p_click(self):
        if self.gl_image is None:
            return
        if self.Base2Poss_tra is None or self.Base2Poss_rot is None:
            return
        if self.Poss_np is None:
            return
        
        rows_to_sel = []
        for row in range(self.model.rowCount()):
            item = self.model.item(row, 0)
            if item.checkState() == Qt.Checked:
                rows_to_sel.append(row)
        if len(rows_to_sel) != 3 :
            msg = QMessageBox(self)
            msg.setWindowTitle("注意")
            msg.setText("3点のみ選択してください。")
            x = msg.exec_()
            return
        base_cam = np.zeros([len(rows_to_sel),5])
        i = 0
        for row in reversed(rows_to_sel):
            base_cam[i,0:3] = self.Poss_np[int(self.model.item(row,4).text()),:]
            base_cam[i,2] = -base_cam[i,2]
            base_cam[i,3] = int(self.model.item(row,5).text())
            base_cam[i,4] = int(self.model.item(row,6).text())
            i = i + 1

        cam_pos , cam_rot , f_cam_pos = self.calcSimpleP3P(base_cam,self.mtx_data)
        selected_value = int(self.comboBox_p3p.currentText()) - 1
        p = selected_value
        invz = np.array([[1,0,0],[0,1,0],[0,0,-1]])
        # rot_c = cam_rot[p,:,:]
        rot_c = invz@cam_rot[p,:,:]@invz
        pos_c = cam_pos[p,:]
        pos_c[2] = -pos_c[2]

        dist = np.array([0,0,self.gl_widget.opts['distance']])
        c0 = pos_c - dist@rot_c
        C = QVector3D(c0[0],c0[1],c0[2])
        self.gl_widget.opts['center'] = C

        rot_q = QQuaternion.toRotationMatrix(self.gl_widget.opts['rotation'])
        # rot_c = rot_c.T
        rot_q[0,0] = rot_c[0,0]
        rot_q[0,1] = rot_c[0,1]
        rot_q[0,2] = rot_c[0,2]
        rot_q[1,0] = rot_c[1,0]
        rot_q[1,1] = rot_c[1,1]
        rot_q[1,2] = rot_c[1,2]
        rot_q[2,0] = rot_c[2,0]
        rot_q[2,1] = rot_c[2,1] 
        rot_q[2,2] = rot_c[2,2]
    
        self.gl_widget.opts['rotation'] = QQuaternion.fromRotationMatrix(rot_q)
        self.gl_widget.update()
        self.set_imagebackground()
        eu = QQuaternion.toEulerAngles(self.gl_widget.opts['rotation'])

        print(C)
        print(cam_pos)
        print(eu)
        print(self.gl_widget.opts['fov'])    
        
    
    def global_rot(self):
        self.camrot = np.eye(3)
        self.campos = np.array([-5,0,20])
        self.camrot = self.rot_zyx(0/180.0*np.pi,30/180.0*np.pi,0/180.0*np.pi)
        
        dist = np.array([0,0,self.gl_widget.opts['distance']])
        print(dist)
        c0 = self.campos - dist@self.camrot
        c0[1] = -c0[1]
        c0_b = self.gl_widget.opts['center']
        c0_b[0] = c0[0]
        c0_b[1] = c0[1]
        c0_b[2] = c0[2]
        self.gl_widget.opts['center'] = c0_b
        rot_c = np.zeros([3,3])
        invz = np.array([[1,0,0],[0,-1,0],[0,0,1]])
        rot_c = invz@self.camrot@invz
        rot_q = QQuaternion.toRotationMatrix(self.gl_widget.opts['rotation'])
        rot_q[0,0] = rot_c[0,0]
        rot_q[0,1] = rot_c[0,1]
        rot_q[0,2] = rot_c[0,2]
        rot_q[1,0] = rot_c[1,0]
        rot_q[1,1] = rot_c[1,1]
        rot_q[1,2] = rot_c[1,2]
        rot_q[2,0] = rot_c[2,0]
        rot_q[2,1] = rot_c[2,1]
        rot_q[2,2] = rot_c[2,2]

        self.gl_widget.opts['rotation'] = QQuaternion.fromRotationMatrix(rot_q)
        print(c0)
        print(c0+dist@rot_c)


    def rotx(self,theta_x):
        rot_x = np.array([[ 1,                 0,                  0],
                    [ 0, np.cos(theta_x), -np.sin(theta_x)],
                    [ 0, np.sin(theta_x),  np.cos(theta_x)]])
        return rot_x

    def roty(self,theta_y):
        rot_y = np.array([[ np.cos(theta_y), 0,  np.sin(theta_y)],
                    [                 0, 1,                  0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
        return rot_y

    def rotz(self,theta_z):
        rot_z = np.array([[ np.cos(theta_z), -np.sin(theta_z), 0],
                    [ np.sin(theta_z),  np.cos(theta_z), 0],
                    [                 0,                  0, 1]])
        return rot_z
    def calcSimpleP3P(self,base_cam,mtx):

        fx = mtx[0,0]
        fy = mtx[1,1]
        Cu = mtx[0,2] * 0.0 + self.gl_image.shape[1]/2.0
        Cv = mtx[1,2] * 0.0 + self.gl_image.shape[0]/2.0
        # Cu = mtx[0,2]
        # Cv = mtx[1,2]
        # print([fx,fy])
        # print([Cu,Cv])
        pixel = np.zeros([3,2],dtype="float32")
        pos = np.zeros([3,3],dtype="float32")

        pixel = base_cam[:,3:5]#画像特徴点位置の切り出し
        pos = base_cam[:,0:3]#三次元位置の切り出し
        q = np.zeros([len(pixel),3])
        for i in range(len(pixel)):
            q[i,0] = (pixel[i,0] - Cu)/fx
            q[i,1] = (pixel[i,1] - Cv)/fy
            q[i,2] = 1
            q[i,:] = q[i,:]/np.linalg.norm(q[i,:])
        a = np.linalg.norm(pos[0,:]-pos[1,:])
        b = np.linalg.norm(pos[1,:]-pos[2,:])
        c = np.linalg.norm(pos[2,:]-pos[0,:])
        ca = q[0,:]@q[1,:]
        cb = q[1,:]@q[2,:]
        cc = q[2,:]@q[0,:]
        ca2 = ca*ca
        cb2 = cb*cb
        cc2 = cc*cc
        B4 = 4*b**2*c**2*ca2 - (a**2-b**2-c**2)**2
        B3 = -4*c**2*(a**2+b**2-c**2)*ca*cb - 8*b**2*c**2*ca2*cc + 4*(a**2-b**2-c**2)*(a**2-b**2)*cc
        B2 = 4*c**2*(a**2-c**2)*cb2 + 8*c**2*(a**2+b**2)*ca*cb*cc + 4*c**2*(b**2-c**2)*ca2 - 2*(a**2-b**2-c**2)*(a**2-b**2+c**2) - 4*(a**2-b**2)**2*cc2
        B1 = -8*a**2*c**2*cb2*cc - 4*c**2*(b**2-c**2)*ca*cb - 4*a**2*c**2*ca*cb + 4*(a**2-b**2)*(a**2-b**2+c**2)*cc
        B0 = 4*a**2*c**2*cb2 - (a**2-b**2+c**2)**2

        s = self.SolveQuarticEquation(B4,B3,B2,B1,B0)
        cam_pos = np.zeros([4,3],dtype="float32")
        cam_rot = np.zeros([4,3,3],dtype="float32")
        f_cam_pos = np.zeros([4,1])
        ans_cnt = 0

        # print(len(s))
        for i in range(len(s)):
            if np.abs(s[i].imag) < 0.00000001:
                u = s[i].real
                v = -((a**2-b**2-c**2)*u**2+2*(b**2-a**2)*cc*u+(a**2-b**2+c**2))/(2*c**2*(ca*u-cb))
                x = a * a / (u * u + v * v - 2.0 * u * v * ca)
                y = b * b / (1 + v * v - 2.0 * v * cb)
                z = c * c / (1 + u * u - 2.0 * u * cc)
                if (x > 0.0):
                    s1 = np.sqrt(x)
                    s2 = u * s1
                    s3 = v * s1
                    P1 = q[0,:]*s2
                    P2 = q[1,:]*s3
                    P3 = q[2,:]*s1
                    P12 = P2-P1
                    P13 = P3-P1
                    P23 = P3-P2
                    zv = np.array(np.cross(P12,P13),dtype="float32")
                    zv = zv/np.linalg.norm(zv)
                    xv = np.array(P12,dtype="float32")
                    xv = xv/np.linalg.norm(xv)
                    yv = np.array(np.cross(xv,zv),dtype="float32")
                    yv = yv/np.linalg.norm(yv)
                    Rotc = np.zeros([3,3])
                    Rotc[:,0] = xv
                    Rotc[:,1] = yv
                    Rotc[:,2] = zv
                    P12b = pos[1,:]-pos[0,:]
                    P13b = pos[2,:]-pos[0,:]
                    zb = np.array(np.cross(P12b,P13b),dtype="float32")
                    zb = zb/np.linalg.norm(zb)
                    xb = np.array(P12b,dtype="float32")
                    xb = xb/np.linalg.norm(xb)
                    yb = np.array(np.cross(xb,zb),dtype="float32")
                    yb = yb/np.linalg.norm(yb)
                    Rota = np.zeros([3,3])
                    Rota[:,0] = xb
                    Rota[:,1] = yb
                    Rota[:,2] = zb
                    xyz1 = P1
                    xyz2 = P2
                    xyz3 = P3
                    t = Rota@Rotc.T@(-P1.T)+pos[0,:].T
                    R = (Rota@Rotc.T)
                    # t = -R.T @ t
                    y_axis = R@np.array([0,1,0])
                    cam_pos[ans_cnt ,:] = t.copy().T
                    cam_rot[ans_cnt,:,:]= R.copy().T
                    # if y_axis[2] < 0:
                    #     f_cam_pos[ans_cnt ,0] = 1
                    ans_cnt = ans_cnt + 1

        return cam_pos , cam_rot , f_cam_pos


    def SolveCubicEquation(self,a,b,c,d):
        if a == 0.0:
            print("Error:a = 0.0\n")
            print("This equation is NOT Cubic.\n")
            return 0
        else:
            A = b/a
            B = c/a
            C = d/a
            p = B-A*A/3.0
            q = 2.0*A*A*A/27.0-A*B/3.0+C
            D = q*q/4.0+p*p*p/27.0
            if D < 0.0:
                x = np.zeros(3)
                theta = np.arctan2(np.sqrt(-D),-q*0.5)
                x[0] = 2.0*np.sqrt(-p/3.0)*np.cos(theta/3.0)-A/3.0
                x[1] = 2.0*np.sqrt(-p/3.0)*np.cos((theta+2.0*np.pi)/3.0)-A/3.0
                x[2] = 2.0*np.sqrt(-p/3.0)*np.cos((theta+4.0*np.pi)/3.0)-A/3.0
            else:
                x = np.zeros(3,dtype=complex)
                u = self.Cuberoot(-q*0.5+np.sqrt(D))
                v = self.Cuberoot(-q*0.5-np.sqrt(D))
                x[0] = u+v-A/3.0
                x[1] = complex(-0.5*(u+v)-A/3.0,np.sqrt(3.0)*0.5*(u-v))
                x[2] = complex(-0.5*(u+v)-A/3.0,-np.sqrt(3.0)*0.5*(u-v))
            return x

    def SolveQuarticEquation(self,a,b,c,d,e):
        if a == 0.0:
            print("Error:a = 0.0\n")
            print("This equation is NOT Quartic.\n")
            return 0
        else:
            x = np.zeros(4,dtype=complex)
            A = b/a
            B = c/a
            C = d/a
            D = e/a
            p = -6.0*(A/4.0)**2.0+B
            q = 8.0*(A/4.0)**3.0-2.0*B*A/4.0+C
            r = -3.0*(A/4.0)**4.0+B*(A/4.0)**2.0-C*A/4.0+D
            t_temp = self.SolveCubicEquation(1.0,-p,-4.0*r,4.0*p*r-q*q)
            t = t_temp[0].real
            m = self.Squreroot(t-p)
            x[0] = (-m+self.Squreroot(-t-p+2.0*q/m))*0.5-A/4.0
            x[1] = (-m-self.Squreroot(-t-p+2.0*q/m))*0.5-A/4.0
            x[2] = (m+self.Squreroot(-t-p-2.0*q/m))*0.5-A/4.0
            x[3] = (m-self.Squreroot(-t-p-2.0*q/m))*0.5-A/4.0
            return x

    def Cuberoot(self,x):
        if x > 0.0:
            return (x)**(1.0/3.0)
        else:
            return -(-x)**(1.0/3.0)
    def Squreroot(self,x):
        r = np.sqrt(x.real*x.real+x.imag*x.imag)
        theta = np.arctan2(x.imag,x.real)
        if x.imag == 0.0:
            if x.real > 0.0:
                y = np.sqrt(r)
            else:
                y = complex(0,np.sqrt(r))
        else:
            if theta < 0.0:
                theta = theta + 2.0*np.pi
            y = complex(np.sqrt(r)*np.cos(theta*0.5),np.sqrt(r)*np.sin(theta*0.5))
        return y


    def on_button_seldel_click(self):
        # 選択した行を削除
        selected_indexes = self.tableView_main.selectionModel().selectedRows()
        for index in reversed(selected_indexes):
            self.model.removeRow(index.row())

    def scatter_rotx(self,theta_x):
        rot_x = np.array([[ 1,                 0,                  0],
                    [ 0, np.cos(theta_x), -np.sin(theta_x)],
                    [ 0, np.sin(theta_x),  np.cos(theta_x)]])
        self.Base2Poss_rot = self.Base2Poss_rot @ rot_x
    def scatter_roty(self,theta_y):
        rot_y = np.array([[ np.cos(theta_y), 0,  np.sin(theta_y)],
                    [                 0, 1,                  0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
        self.Base2Poss_rot = self.Base2Poss_rot @ rot_y

    def scatter_rotz(self,theta_z):
        rot_z = np.array([[ np.cos(theta_z), -np.sin(theta_z), 0],
                    [ np.sin(theta_z),  np.cos(theta_z), 0],
                    [                 0,                  0, 1]])
        self.Base2Poss_rot = self.Base2Poss_rot @ rot_z
    def scatter_trax(self,tra_x):
        tra_x = np.array([tra_x,0,0])
        self.Base2Poss_tra = self.Base2Poss_tra + tra_x@self.Base2Poss_rot
    def scatter_tray(self,tra_y):
        tra_y = np.array([0,tra_y,0])
        self.Base2Poss_tra = self.Base2Poss_tra + tra_y@self.Base2Poss_rot
    def scatter_traz(self,tra_z):
        tra_z = np.array([0,0,tra_z])
        self.Base2Poss_tra = self.Base2Poss_tra + tra_z@self.Base2Poss_rot

    def reset_rottra(self):
        self.Base2Poss_rot = np.eye(3)
        self.Base2Poss_tra = np.zeros([1,3])

    def update_pos(self):
        if self.Plot is None:
            self.Plot = gl.GLScatterPlotItem()

        if self.Base_Poss_np is None:
            return
        self.Poss_np = (self.Base_Poss_np + self.Base2Poss_tra)@self.Base2Poss_rot
        # self.Poss_np = self.Base2Poss_tra + self.Base_Poss_np@self.Base2Poss_rot
        self.Plot.setData(pos = self.Poss_np)

        self.update_selpos()

    def update_selpos(self):
        if self.Plotr is None:
            self.Plotr = gl.GLScatterPlotItem()
        else:
            for item in self.gl_widget.items:
                if self.Plotr == item:
                    self.gl_widget.removeItem(self.Plotr)
                    self.Plotr = gl.GLScatterPlotItem()
                    break
        if self.selpos_index is None:
            return
        if len(self.selpos_index) == 0:
            return
        poss = np.zeros([len(self.selpos_index),3])
        cols = np.zeros([len(self.selpos_index),4])
        for i in range(len(self.selpos_index)):
            poss[i,:] = self.Poss_np[self.selpos_index[i],:]
            cols[i,:] = np.array([1,0,0,1])

        self.Plotr.setData(pos=poss, size=self.slider_3dpointsize.value()/100.0*10.5, color=cols, pxMode=False)
        self.gl_widget.addItem(self.Plotr)
        self.set_imagebackground()
class ClickablePlotWidget(gl.GLViewWidget):

    task1 = None
    task2 = None
    task3 = None
    task4 = None
    task5 = None
    task6 = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.task1(event)
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.task2(event)
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.task3(event)
    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.task4(event)
    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.task5(event)
    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        self.task6(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = PointCloudViewer()

    sys.exit(app.exec_())
