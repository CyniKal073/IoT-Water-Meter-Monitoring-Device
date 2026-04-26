import onnxruntime as ort
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import cv2

YOLO_MODEL = "models/detect/watermeter_det_v11s.pt"
OCR_REC_MODEL = "models/ppocr4_rec"
METER_READZONE_SIZE = (300, 75)


class MeterTool:
    model = None
    ocr = None

    def __init__(self, yoloModel=YOLO_MODEL, ocrModel=OCR_REC_MODEL, METER_READZONE_SIZE=METER_READZONE_SIZE):
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


if __name__ == "__main__":
    pass
