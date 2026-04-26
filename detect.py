import metertool
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import onnxdet
import ortdet
import socket

conf_threshold = 0.5  # 置信度阈值
iou_threshold = 0.5  # IoU阈值，用于旋转NMS
imgsz = (640, 640)  # 模型输入大小

YOLO_MODEL = "models/detect/watermeter_det_v11s.onnx"
OCR_REC_MODEL = "models/ppocr4_rec"
METER_READZONE_SIZE = (300,75)

'''
capture = cv2.VideoCapture(0)
imgobj = cv2.imread('pic.png')
while True:
    ret, frame = capture.read()
    #frame = cv2.flip(frame, 1)
    cv2.imshow('video', frame)
    x = cv2.waitKey(30)
    if x == ord('s'):
        cv2.destroyWindow('img')
        imgobj = onnxdet.process_images_in_folder(frame, YOLO_MODEL, conf_threshold, iou_threshold, imgsz)
    cv2.imshow('img', imgobj)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
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

wm = ortdet.MeterTool(YOLO_MODEL, OCR_REC_MODEL, METER_READZONE_SIZE)

capture = cv2.VideoCapture(0)
imgobj = cv2.imread('pic.png')
while True:
    ret, frame = capture.read()
    cv2.imshow('video', frame)
    UDP_PIC_Send(frame)
    x = cv2.waitKey(30)
    if x == ord('s'):
        cv2.destroyWindow('img')
        imgobj = wm.cropMeterReadZone(frame)
        npImg = np.array(imgobj)
        res = wm.recognizeMeterRead(npImg)
        imgobj = cv2.cvtColor(npImg, cv2.COLOR_RGB2BGR)
        res = 'data' + res
        UDP_Data_Send(res)
    cv2.imshow('img', imgobj)


cv2.waitKey(0)
cv2.destroyAllWindows()

