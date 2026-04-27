from ultralytics import YOLO

model = YOLO(r"C:\Users\a511009\Desktop\Projeto\PPE-Detection\runs\detect\train\weights\best.pt")

results = model.predict(
    source="teste.jpg",
    conf=0.25,
    save=True
)