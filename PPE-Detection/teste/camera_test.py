from ultralytics import YOLO
import cv2

# Carrega o modelo treinado
model = YOLO(r"C:\Users\a511009\Desktop\Projeto\PPE-Detection\runs\detect\train\weights\best.pt")

# Abre a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar imagem da câmera")
        break


    # Aplica o modelo no frame
    results = model(frame, conf=0.5, classes=[0, 2, 6])

    # Desenha as caixas na imagem
    annotated_frame = results[0].plot()

    # Mostra a imagem com as detecções
    cv2.imshow("Detecção de EPIs", annotated_frame)

    # Pressione Q para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()