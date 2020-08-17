# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import threading
from threading import Thread
import numpy
import argparse
import torch
import torch.backends.cudnn as cudnn


from utils.datasets import *
from utils.utils_xy import *
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.experimental import attempt_load


from PyQt5.QtGui import QPixmap, QImage,  QMovie,  QFont, QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
import cv2

show_image_size = [760, 570]

class Ui_MainWindow(object):
    
    def __init__(self):
        super().__init__()
        self.flag_stop = True
        self.flag_load_model = True
        self.flag_pause = False
        self.source = "0"
        self.flag_object_detect = True
        self.flag_video_path_change = False
    
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(961, 737)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.FaceToShowNs = QtWidgets.QLabel(self.centralwidget)
        self.FaceToShowNs.setGeometry(QtCore.QRect(20, 600, 915, 80))
        self.FaceToShowNs.setText("")
        self.FaceToShowNs.setPixmap(QtGui.QPixmap("UI_images/gray.jpg"))
        self.FaceToShowNs.setObjectName("FaceToShowNs")
        
        self.MainShow = QtWidgets.QLabel(self.centralwidget)
        self.MainShow.setGeometry(QtCore.QRect(20, 20, 760, 570))
        self.MainShow.setText("")
        self.MainShow.setPixmap(QtGui.QPixmap("UI_images/weapon_rpg.jpg"))
        self.MainShow.setScaledContents(True)
        self.MainShow.setObjectName("MainShow")
        
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(790, 20, 145, 120))
        
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 46, 48))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 46, 48))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 46, 48))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(47, 46, 48))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        
        self.logo.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Roman")
        font.setPointSize(30)
        self.logo.setFont(font)
        self.logo.setAutoFillBackground(True)
        self.logo.setTextFormat(QtCore.Qt.AutoText)
        self.logo.setAlignment(QtCore.Qt.AlignCenter)
        self.logo.setObjectName("logo")
        
        self.picture = QtWidgets.QRadioButton(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(60, 613, 116, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.picture.setFont(font)
        self.picture.setObjectName("picture")
        
        self.LocalCam = QtWidgets.QRadioButton(self.centralwidget)
        self.LocalCam.setGeometry(QtCore.QRect(60, 643, 116, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.LocalCam.setFont(font)
        self.LocalCam.setObjectName("LocalCam")
        
        self.Start = QtWidgets.QPushButton(self.centralwidget)
        self.Start.setGeometry(QtCore.QRect(208, 620, 130, 40))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.Start.setFont(font)
        self.Start.clicked.connect(self.slots_target_btnstate)
        self.Start.setCheckable(True)
        self.Start.setObjectName("Start")
        
        self.Pause = QtWidgets.QPushButton(self.centralwidget)
        self.Pause.setGeometry(QtCore.QRect(378, 620, 130, 40))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.Pause.setFont(font)
        self.Pause.setObjectName("Pause")
        
        self.WeaponDetectionFlag = QtWidgets.QPushButton(self.centralwidget)
        self.WeaponDetectionFlag.setGeometry(QtCore.QRect(548, 620, 130, 40))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.WeaponDetectionFlag.setFont(font)
        self.WeaponDetectionFlag.setObjectName("WeaponDetectionFlag")
        
        self.UploadWeaponDatabase = QtWidgets.QPushButton(self.centralwidget)
        self.UploadWeaponDatabase.setGeometry(QtCore.QRect(718, 620, 130, 40))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.UploadWeaponDatabase.setFont(font)
        self.UploadWeaponDatabase.setCheckable(True)
        self.UploadWeaponDatabase.setObjectName("UploadWeaponDatabase")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(790, 160, 145, 30))
        font = QtGui.QFont()
        font.setFamily("Roman")
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        self.label_result_box = QtWidgets.QLabel(self.centralwidget)
        self.label_result_box.setGeometry(QtCore.QRect(790, 190, 151, 400))
        self.label_result_box.setStyleSheet("border:1px solid black")
        self.label_result_box.setText("")
        self.label_result_box.setObjectName("label_result_box")
        
        self.label_object_tip = QtWidgets.QLabel(self.centralwidget)
        self.label_object_tip.setGeometry(QtCore.QRect(790, 195, 145, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        self.label_object_tip.setFont(font)
        self.label_object_tip.setObjectName("label_object_tip")
        
        self.label_num_tip = QtWidgets.QLabel(self.centralwidget)
        self.label_num_tip.setGeometry(QtCore.QRect(880, 195, 55, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(13)
        self.label_num_tip.setFont(font)
        self.label_num_tip.setObjectName("label_num_tip")
        
        self.label_result_object = QtWidgets.QLabel(self.centralwidget)
        self.label_result_object.setGeometry(QtCore.QRect(790, 220, 145, 370))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label_result_object.setFont(font)
        self.label_result_object.setText("")
        self.label_result_object.setObjectName("label_result_object")
        
        self.label_result_num = QtWidgets.QLabel(self.centralwidget)
        self.label_result_num.setGeometry(QtCore.QRect(880, 220, 55, 370))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label_result_num.setFont(font)
        self.label_result_num.setText("")
        self.label_result_num.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result_num.setObjectName("label_result_num")
        
        self.pictureName = QtWidgets.QTextEdit(self.centralwidget)
        self.pictureName.setGeometry(QtCore.QRect(200, 660, 161, 21))
        self.pictureName.setObjectName("pictureName")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 961, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "武器检测"))
        self.logo.setText(_translate("MainWindow", "<html><head/><body><p>武器</p><p>检测</p><p><br/></p></body></html>"))
        self.picture.setText(_translate("MainWindow", "图片检测"))
        self.LocalCam.setText(_translate("MainWindow", "本地摄像头"))
        self.Start.setText(_translate("MainWindow", "开始"))
        self.Pause.setText(_translate("MainWindow", "暂停"))
        self.WeaponDetectionFlag.setText(_translate("MainWindow", "检测"))
        self.UploadWeaponDatabase.setText(_translate("MainWindow", "停止"))
        self.label.setText(_translate("MainWindow", "目标检测结果"))
        self.label_object_tip.setText(_translate("MainWindow", " 目标"))
        self.label_num_tip.setText(_translate("MainWindow", "数量"))
        self.pictureName.textChanged.connect(self.show_text_func)  # 1



    def show_text_func(self):
        self.source = self.pictureName.toPlainText()
        
    
    def slots_target_btnstate(self):
                #################################不检测摄像头是否连接好#################################
        if self.Start.isChecked():

            print("pushButton")
            self.Start.setCheckable(False)
            self.Pause.setCheckable(True)
            self.WeaponDetectionFlag.setCheckable(True)
            self.UploadWeaponDatabase.setCheckable(True)
            if self.flag_stop:
                print("Start")
                self.flag_stop = False
                if self.flag_load_model:
                    self.clickStart()

            else:
                print("Pause")
                self.flag_pause = False

    def clickStart(self):      
        
        self.thread = Thread_clickStart(self.MainShow,self.source)
        # 连接信号
        self.thread._signal.connect(self.call_backlog)  # 进程连接回传到GUI的事件
        # 开始线程
        self.thread.start()

    def call_backlog(self,msg):
        
        label_result_object_text = ""
        label_result_num_text = ""
        
        label_result_object_text += "    " + "武器" + "\n"
        # label_result_num_text += "    " + str(detect_weapons_num) + "\n"
        
        self.label_result_object.setText(label_result_object_text)
        self.label_result_num.setText(label_result_num_text)

        im0 = cv2.cvtColor(msg, cv2.COLOR_RGB2BGR)
        im0 = QImage(im0.data, im0.shape[1], im0.shape[0], QImage.Format_RGB888)
        self.MainShow.setPixmap(QPixmap.fromImage(im0.scaled(show_image_size[0],
                                                             show_image_size[1])))

       
        
        
class Thread_clickStart(QThread):  # 线程1
    _signal = pyqtSignal(numpy.ndarray)
    def __init__(self,MainShow,source):
        super().__init__()
        self.MainShow = MainShow
        self.source = source 
        
    def run(self):
        self.MainShow.setPixmap(QtGui.QPixmap("UI_images/loading_pic.jpg"))
        cv2.waitKey(1)

        self.detect(source = self.source)       

    def detect(self,save_img=False,source = "0"):
        out, weights, view_img, save_txt, imgsz = \
            opt.output, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    
        # Initialize
        device = select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA
    
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16
    
        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()
    
        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)
    
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
    
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
    
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
    
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
    
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s
    
                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
    
                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
    
                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    
                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))
    
                # Stream results
                if view_img:
                    self._signal.emit(im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration 
    

        print('Done. (%.3fs)' % (time.time() - t0))



if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/exp15/weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    # opt.img_size = check_img_size(opt.img_size)    
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
