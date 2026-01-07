import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'')

    model.train(
        data=r"E:\Desktop\YOLOv8\mydataS.yaml",

                cache=False,
                imgsz=640,
                epochs=200,
                single_cls=False,
                batch=4,
                close_mosaic=15,
                cos_lr=True,
                workers=0,
                device='0',
                optimizer='SGD',

                amp=False,
                project='runs/train',
                name='exp',
                )
