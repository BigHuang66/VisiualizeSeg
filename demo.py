import sys, json
import os, cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal
from ui import Ui_MainWindow
import numpy as np

""" paddle """
import paddle
from paddleseg import utils
from paddleseg.core import infer
import paddleseg.transforms as T
from paddleseg.datasets import Dataset
from paddleseg.core import predict
from paddleseg.models import UNet
# from paddleseg.models.

# 加载神经网络的线程， 防止UI主线程卡顿
class NetworkThread(QThread):
    trigger = pyqtSignal()
    def __init__(self):
        super(NetworkThread, self).__init__()
    # 线程开启
    def run(self):
        global model, transforms
        model, transforms = networkInit()
        self.trigger.emit()

# 推理时耗时操作的多线程，防止主UI卡顿
class InferThread(QThread):
    infer_request = pyqtSignal(QPixmap)
    def __init__(self):
        super(InferThread, self).__init__()
    def run(self):
        global infer_img_path, model, transforms

        model_path = "./models/Unet/model.pdparams"
        result = predict(model, transforms, im_path=infer_img_path)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        height, width = result.shape[:2]
        result = QImage(result, width, height, width*3, QImage.Format_RGB888)
        result = QPixmap.fromImage(result)
        self.infer_request.emit(result)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.backEnd(self)
       
        self.fileList = []
        self.maskList = []
        self.ui.run_button.setEnabled(False)
        self.ui.open_fold.setEnabled(False)
        self.ui.open_mask.setEnabled(False)
        self.ui.pre_img_button.setEnabled(False)
        self.ui.next_img_button.setEnabled(False)
        self.ui.switch_show.setEnabled(False)
        self.refresh_model_library()
    
    # 软件打开时自动扫描models下所有可用模型
    def refresh_model_library(self):
        models = os.listdir('models')
        for model in models:
            if len(os.listdir(os.path.join('models', model))) != 0:
                self.ui.model_list.addItem(model.split('/')[-1].split('.')[0])

    # 选择文件，暂且不支持文件夹打开模式
    def openFold(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Open file', '.', "Image files (*.jpg *.tif *.bmp)")
        self.fileList = []
        self.ui.img_list.clear()
        self.ui.run_button.setEnabled(False)
        for name in fnames:
            self.ui.img_list.addItem(name.split('/')[-1])
            self.fileList.append(name)
        if len(self.fileList) > 0:
            self.ui.input_show.setPixmap(QPixmap(self.fileList[0]))
            self.ui.run_button.setEnabled(True)
            self.ui.pre_img_button.setEnabled(True)
            self.ui.next_img_button.setEnabled(True)

    def openMask(self):
        pass

    # 读取文件下拉列表的文件，默认显示首文件，当选择其他文件时，自动调用此函数实现显示切换
    # 只依靠文件顺序索引进行选择，并不涉及文件名称
    def chooseFile(self):
        index = self.ui.img_list.currentIndex()
        self.ui.input_show.setPixmap(QPixmap(self.fileList[index]))
    # 前一张
    def priorImage(self):
        index = self.ui.img_list.currentIndex()
        if index == 0:
            return
        index -= 1
        self.ui.img_list.setCurrentIndex(index)
        self.ui.input_show.setPixmap(QPixmap(self.fileList[index]))
    # 下一张
    def nextImage(self):
        index = self.ui.img_list.currentIndex()
        if index == len(self.fileList)-1:
            return
        index += 1
        self.ui.img_list.setCurrentIndex(index)
        self.ui.input_show.setPixmap(QPixmap(self.fileList[index]))

    # 当模型旋转下拉框切换时，调用此函数，线程里切换model，同时对按键进行使能使能处理
    def chooseModel(self):
        win.ui.open_fold.setEnabled(False)
        win.ui.run_button.setEnabled(False)
        self.networkThread = NetworkThread()
        self.networkThread.trigger.connect(networkInitOK)
        self.networkThread.start()

    # 执行Run按键
    def inference(self):
        global infer_img_path
        
        index = self.ui.img_list.currentIndex()
        infer_img_path = self.fileList[index]
        self.infer = InferThread()
        self.infer.infer_request.connect(self.show_infer_result)
        self.infer.start()
        model_path = "./models/Unet/model.pdparams"

        # result = predict(model, transforms, im_path=self.fileList[index])
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # height, width = result.shape[:2]
        # result = QImage(result, width, height, width*3, QImage.Format_RGB888)
        # result = QPixmap.fromImage(result)
        # self.ui.output_show.setPixmap(QPixmap(result))
        # #result = '真实类别:{}           预测类别: {}'.format(label, class_names[predict.data.cpu().numpy().tolist()[0]])
        # result = "诊断完成!  image shape: " + str(img.shape) + " ,mask shape: " + str(height) + " " + str(width)
        # self.ui.result_show.setText(result)
    def show_infer_result(self, result):
        self.ui.output_show.setPixmap(result)
        result = "诊断完成!  "
        self.ui.result_show.setText(result)
    
# 切换前失能按键  网络切换完成后 使能按键
def networkInitOK():
    win.ui.open_fold.setEnabled(True)
    if len(win.fileList) > 0:
        win.ui.run_button.setEnabled(True)

def predict(model, transforms, im_path):
    """
    predict and visualize the image_list.
    """
    
    color_map = {'1': [180, 105, 255],   # HotPink
                '2': [255, 0, 0],     # Magenta [255, 0, 255]
                '3': [0, 0, 255],       # red
                '4': [255, 0, 0]}       # blue

    with paddle.no_grad():
        im = cv2.imread(im_path)
        im = cv2.resize(im, (512, 512))
        image = im.copy()
        im, _ = transforms(im)
        im = im[np.newaxis, ...]
        im = paddle.to_tensor(im)

        output = model(im)[0]
        output = output.numpy()
        output = np.argmax(output, axis=1)
        output = output.transpose(1,2,0).astype('uint8')
        output = output.squeeze()
        for i in range(1, 3):
            mask = (output == i).astype(np.bool)
            color_mask = np.array(color_map[str(i)], dtype=np.uint8)
            image[mask] = image[mask] * 0.5 + color_mask * 0.5
    return image
 


# 网络模型的初始化  线程 networkThread 中调用，开启软件或切换model时
def networkInit():

    transforms = T.Compose([
    # T.Resize(target_size=(512, 512)),
    T.Normalize()
    ])

    model_name = win.ui.model_list.currentText()
    models_dict = './models'
    model_path = os.path.join(models_dict, model_name, 'model.pdparams')
    
    model = UNet(num_classes=3)
    model.set_dict(paddle.load(model_path))
    model.eval()
    
    return model, transforms




if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    win = MainWindow()
    # 创建多线程
    networkThread = NetworkThread()
    networkThread.trigger.connect(networkInitOK)
    networkThread.start()

    
    # model_ft, transform = networkInit()

    win.show()

    
    sys.exit(app.exec_())

