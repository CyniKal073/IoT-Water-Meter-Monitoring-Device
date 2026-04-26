from ultralytics import YOLO
from PIL import Image
import numpy
from paddleocr import PaddleOCR
import cv2

YOLO_MODEL = "models/detect/watermeter_det_v11s.pt"
OCR_REC_MODEL = "models/ppocr4_rec"
METER_READZONE_SIZE = (300,75)

class MeterTool:
    model=None
    ocr=None

    def __init__(self,yoloModel=YOLO_MODEL,ocrModel=OCR_REC_MODEL,METER_READZONE_SIZE=METER_READZONE_SIZE):
        self.model = YOLO(yoloModel)
        self.ocr = PaddleOCR(rec_model_dir=ocrModel, use_angle_cls=True)
        self.size = METER_READZONE_SIZE

    def cropMeterReadZone(self,srcImg):
        #srcImg = Image.open(imagePath)
        srcImg = Image.fromarray(cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB))
        res = self.model.predict(source=srcImg)
        if len(res) < 1:
            return None		# No object detected
        obbInfo = res[0].obb	# Only take the first image's info
        print(obbInfo)
        clsList = obbInfo.cls.tolist()
        xy4 = []
        for i in range(len(obbInfo)):
            if int(clsList[i]) == 0:  # The main target
                xy4 = obbInfo.xyxyxyxy.tolist()[i]
        if xy4 == []:
            return None     # Unexpected
        if xy4[0][0] >= xy4[2][0]:
            data = (
                xy4[2][0],xy4[2][1],
                xy4[3][0],xy4[3][1],
                xy4[0][0],xy4[0][1],
                xy4[1][0],xy4[1][1]
                                    )
        else:
            data = (
                xy4[3][0], xy4[3][1],
                xy4[0][0], xy4[0][1],
                xy4[1][0], xy4[1][1],
                xy4[2][0], xy4[2][1]
                                    )
        return srcImg.transform(self.size, Image.QUAD,
                                data = data,
                                resample = Image.BICUBIC
                                )

    def recognizeMeterRead(self,imgObj):
        ocrResult = self.ocr.ocr(imgObj, det=False, cls=True)[0]
        if len(ocrResult) > 0:
            ocrResult = ocrResult[0]    # Take first result
            return ocrResult[0].replace(" ","")
        else:
            return None

    def getMeterReadVal(self,imagePath):
        img = self.cropMeterReadZone(imagePath)
        if img is None:
            return None
        npImg = numpy.array(img)
        res = self.recognizeMeterRead(npImg)
        npImg = numpy.array(img.rotate(180))
        res2 = self.recognizeMeterRead(npImg)
        return res,res2

if __name__ == "__main__":
    pass
