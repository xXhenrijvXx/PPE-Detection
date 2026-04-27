from ultralytics import YOLO

model = YOLO(r"C:\Users\a511009\Desktop\Projeto\PPE-Detection\runs\detect\train\weights\best.pt")

model.export(format="ncnn")