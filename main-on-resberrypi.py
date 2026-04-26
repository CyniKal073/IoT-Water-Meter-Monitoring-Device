import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import time
import RPi.GPIO as GPIO
from smbus2 import SMBus
from multiprocessing import Process, RawArray, Lock
import socket
from flask import Flask, render_template, Response



class MeterTool:
    model = None
    ocr = None

    def __init__(self, yoloModel, ocrModel, METER_READZONE_SIZE):
        self.model = ort.InferenceSession(
            yoloModel,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else [
                "CPUExecutionProvider"],
        )
        print("YOLO11 目标检测 ONNXRuntime")
        print("模型名称：", yoloModel)
        self.ocr = PaddleOCR(rec_model_dir=ocrModel, use_angle_cls=True)
        self.size = METER_READZONE_SIZE

    def preprocess(self, srcImg):
        self.img = srcImg
        self.img_height, self.img_width = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, (r, r), (dw, dh)

    def cropMeterReadZone(self, srcImg):
        # srcImg = Image.open(imagePath)
        model_inputs = self.model.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        print(f"模型输入尺寸：宽度 = {self.input_width}, 高度 = {self.input_height}")
        img_data = self.preprocess(srcImg)
        output = self.model.run(None, {model_inputs[0].name: img_data})
        output = output[0]
        num_detections = output.shape[2]  # 获取检测的边界框数量
        num_classes = output.shape[1] - 6  # 计算类别数量
        xy4 = []
        for i in range(num_detections):
            detection = output[0, :, i]
            x_center, y_center, width, height = detection[0], detection[1], detection[2], detection[3]  # 提取边界框的中心坐标和宽高
            angle = detection[-1]  # 提取旋转角度
            class_confidences = detection[4:4 + num_classes]  # 获取类别置信度
            if class_confidences.size == 0:
                continue
            class_id = np.argmax(class_confidences)  # 获取置信度最高的类别索引
            confidence = class_confidences[class_id]  # 获取对应的置信度
            ratio = self.ratio
            dwdh = (self.dw, self.dh)
            if class_id == 0 and confidence > 0.5:  # 过滤掉低置信度的检测结果
                x_center = (x_center - dwdh[0]) / ratio[0]  # 还原中心点 x 坐标
                y_center = (y_center - dwdh[1]) / ratio[1]  # 还原中心点 y 坐标
                width /= ratio[0]  # 还原宽度
                height /= ratio[1]  # 还原高度
                cos_angle = np.cos(angle)  # 计算旋转角度的余弦值
                sin_angle = np.sin(angle)  # 计算旋转角度的正弦值
                dx = width / 2  # 计算宽度的一半
                dy = height / 2  # 计算高度的一半
                xy4 = [
                    (int(x_center - cos_angle * dx + sin_angle * dy), int(y_center - sin_angle * dx - cos_angle * dy)),
                    (int(x_center + cos_angle * dx + sin_angle * dy), int(y_center + sin_angle * dx - cos_angle * dy)),
                    (int(x_center + cos_angle * dx - sin_angle * dy), int(y_center + sin_angle * dx + cos_angle * dy)),
                    (int(x_center - cos_angle * dx - sin_angle * dy), int(y_center - sin_angle * dx + cos_angle * dy)),
                    ]

        srcImg = Image.fromarray(cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB))
        if xy4 == []:
            return None  # Unexpected
        if xy4[0][0] >= xy4[2][0]:
            data = (
                xy4[3][0], xy4[3][1],
                xy4[2][0], xy4[2][1],
                xy4[1][0], xy4[1][1],
                xy4[0][0], xy4[0][1]
            )
        else:
            data = (
                xy4[0][0], xy4[0][1],
                xy4[3][0], xy4[3][1],
                xy4[2][0], xy4[2][1],
                xy4[1][0], xy4[1][1]
            )
        return srcImg.transform(self.size, Image.QUAD,
                                data=data,
                                resample=Image.BICUBIC
                                )

    def recognizeMeterRead(self, imgObj):
        ocrResult = self.ocr.ocr(imgObj, det=False, cls=True)[0]
        if len(ocrResult) > 0:
            ocrResult = ocrResult[0]  # Take first result
            return ocrResult[0].replace(" ", "")
        else:
            return None

    def getMeterReadVal(self, imagePath):
        img = self.cropMeterReadZone(imagePath)
        if img is None:
            return None
        npImg = np.array(img)
        res = self.recognizeMeterRead(npImg)
        npImg = np.array(img.rotate(180))
        res2 = self.recognizeMeterRead(npImg)
        return res, res2

def UDP_PIC_Send(frame):
    udp_socket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  #确定通信协议类型
    recv_addr = ('127.0.0.1', 8081)
    img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]  # 对每帧图进行编码,压缩画质
    data = np.array(img_encode)
    byte_encode = data.tobytes()
    udp_socket.sendto(byte_encode, recv_addr)
    udp_socket.close()


def UDP_Data_Send(data):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 确定通信协议类型
    recv_addr = ('127.0.0.1', 8081)
    udp_socket.sendto(data.encode("utf-8"), recv_addr)
    udp_socket.close()

def TempRead(addr):
    i2c = SMBus(1)
    i2c.open(1)
    i = i2c.read_byte(addr, force=None)
    return i

def GPIO_ReadHal():
    if GPIO.input(18):
        return 1  #open
    else:
        return 0 #close

def GPIO_Init(num):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(num, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def PWM_Init(PWM):
    GPIO.setmode(GPIO.BCM)     #设置编号方式
    GPIO.setup(PWM, GPIO.OUT)  #设置 引脚为输出模式

def Servo_On(PWM):
    global Servo_State
    p = GPIO.PWM(PWM, 50)  #将 引脚初始化为PWM实例 ，频率为50Hz
    p.start(0)             #开始脉宽调制，参数范围为： (0.0 <= dc <= 100.0)
    p.ChangeDutyCycle(12.5)
    time.sleep(0.1)
    Servo_State = 1

def Servo_Off(PWM):
    global Servo_State
    p = GPIO.PWM(PWM, 50)  #将 引脚初始化为PWM实例 ，频率为50Hz
    p.start(0)             #开始脉宽调制，参数范围为： (0.0 <= dc <= 100.0)
    p.ChangeDutyCycle(2.5)
    time.sleep(0.1)
    Servo_State = 0
    
def Flask_On():
    global state_list
    global w_data
    global t_data
    global h_data
    state_list = ['关盖', '自动开盖', '人为开盖']
    w_data = '0'
    t_data = '0'
    h_data = '0'
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 确定通信协议类型
    recv_addr = ('127.0.0.1', 8081)
    while True:
        try:
            udp_socket.bind(recv_addr)
            print('emm')
        except OSError:
            pass
        else:
            break
    udp_socket.setblocking(0)  # 设置为非阻塞模式
    
    app = Flask(__name__)
    
    def generate():
        image = cv2.imread('pic.png')
        image = cv2.imencode('.jpg', image)[1].tobytes()
        while True:
            try:
                recv_byte, send_addr = udp_socket.recvfrom(921600)
                receive_data = np.frombuffer(recv_byte, dtype='uint8')
                r_img = cv2.imdecode(receive_data, 1)
                image = cv2.imencode('.jpg', r_img)[1].tobytes()
            except BlockingIOError:
                pass
            except cv2.error:
                try:
                    data = recv_byte.decode()
                    for i in len(data):
                        if data[i] == 'w':
                            j = i
                            while data[j] != 't':
                                j = j + 1
                            w_data = data[i:j]
                        if data[i] == 't':
                            j = i
                            while data[j] != 'h':
                                j = j + 1
                            t_data = data[i:j]
                        if data[i] == 'h':
                            h_data = data[i+1]
                except TypeError:
                    pass    
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')
    
    @app.route('/')
    def main():
        return render_template('test.html', test='hello', water_data=w_data,tem_data=t_data, state_data=state_list[int(h_data)])
    
    @app.route('/img_data')
    def img_data():
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='127.0.0.1', port='23333', debug=True, threaded=True, use_reloader=False)

def Detect():
    global Hal_State
    global Servo_State
    global h_data
    global w_data
    global t_data
    global send_data
    
    Hal_State = 0
    Servo_State = 0
    w_data, t_data, h_data = '0', '0', '0'
    send_data = 'w' + w_data + 't' + t_data + 'h' + 'h_data'

    GPIO_Init(4)
    PWM_Init(18)
    Servo_On(18)

    conf_threshold = 0.5  # 置信度阈值
    iou_threshold = 0.5  # IoU阈值，用于旋转NMS
    imgsz = (640, 640)  # 模型输入大小
    YOLO_MODEL = "models/detect/watermeter_det_v11s.onnx"
    OCR_REC_MODEL = "models/ppocr4_rec"
    METER_READZONE_SIZE = (300,75)
    wm = MeterTool(YOLO_MODEL, OCR_REC_MODEL, METER_READZONE_SIZE)
    imgobj = cv2.imread('pic.png')
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = frame.reshape(480, 640, 3)
        cv2.imshow('video', frame)
        UDP_PIC_Send(frame)
        x = cv2.waitKey(30)
        if x == ord('s'):
             break
        imgori = wm.cropMeterReadZone(frame)
        if imgori != None:
            npImg = np.array(imgori)
            imobj = imgori
            res = wm.recognizeMeterRead(npImg)
            if res != None:
                w_data = res  
            imgobj = cv2.cvtColor(npImg, cv2.COLOR_RGB2BGR)
        t_data = str(TempRead(0x48))
        Hal_State = GPIO_ReadHal()
        if ~(Hal_State ^ Servo_State):
            if Servo_State == 1:
                h_data = '1'
            else:
                h_data = '2'
        else:
            h_date = '0'
    
        send_data = 'w' + w_data + 't' + t_data + 'h' + h_data
        UDP_Data_Send(send_data)
        print(send_data, '\n')
    Servo_Off(18)
    cv2.destroyAllWindows()
        
process_detect = Process(target=Detect)
process_Flask = Process(target=Flask_On)

if __name__ == '__main__':
    process_detect.start()
    process_Flask.start()
    process_detect.join()
    process_Flask.join()


    
