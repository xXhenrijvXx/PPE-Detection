# o código comentado será usado para embarcar na raspberry posteriormente

from ultralytics import YOLO
import cv2
# RASPBERRY
# import RPi.GPIO as GPIO

# Frames quantity in 1 second
QTY = 40
# person class
PERSON_CLASS = 6
# helmet, vest e goggle
PPE_CLASSES = [0, 2, 4]

def is_box_inside(ppe_box, person_box, min_ratio=0.5):


    ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_box

    p_x1, p_y1, p_x2, p_y2 = person_box

    # Centro do EPI
    cx = (ppe_x1 + ppe_x2) / 2
    cy = (ppe_y1 + ppe_y2) / 2

    center_inside = p_x1 <= cx <= p_x2 and p_y1 <= cy <= p_y2

    # Interseção entre caixa do EPI e caixa da pessoa
    ix1 = max(ppe_x1, p_x1)
    iy1 = max(ppe_y1, p_y1)
    ix2 = min(ppe_x2, p_x2)
    iy2 = min(ppe_y2, p_y2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    ppe_area = max(1, (ppe_x2 - ppe_x1) * (ppe_y2 - ppe_y1))

    inside_ratio = inter_area / ppe_area

    return center_inside or inside_ratio >= min_ratio

# RASPBERRY
# LEDs
# GPIO.setmode(GPIO.BCM)
# green_led = 17
# red_led = 27

# GPIO.setup(green_led, GPIO.OUT)
# GPIO.setup(red_led, GPIO.OUT)

# GPIO.output(green_led, False)
# GPIO.output(red_led, False)

# Carrega o modelo treinado
model = YOLO("runs/detect/train/weights/best.pt")

# Abre a webcam
cap = cv2.VideoCapture(0)

counter = 0
frames_status = [2 for i in range(QTY)]

try:
    while True:
        
        ret, frame = cap.read()

        # Status OK (Acender LED VERDE)
        status_frame = 1

        if not ret:
            print("Erro ao capturar imagem da câmera")
            # fazer algo com os leds para informar erro
            break

        # Aplica o modelo no frame
        results = model(frame, conf=0.5, classes=(PPE_CLASSES + [PERSON_CLASS]))
        result = results[0]

        people = []
        ppes = []

        for box in result.boxes:
            class_id = int(box.cls[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = {
                "class_id": class_id,
                "box": [x1, y1, x2, y2]
            }

            if class_id == PERSON_CLASS:
                people.append(detection)
            elif class_id in PPE_CLASSES:
                ppes.append(detection)
        
        # Desenha as caixas na imagem
        annotated_frame = results[0].plot()

        if not people:
            print("NO PERSON DETECTED")  
            status_frame = None
            # GPIO.output(green_led, False)
            # GPIO.output(red_led, False)
        else:
                
            for person in people:

                if status_frame == 0:
                    break

                person_box = person["box"]

                person_ppe = []

                for ppe in ppes:
                    ppe_box = ppe["box"]
                    if is_box_inside(ppe_box, person_box):
                        person_ppe.append(ppe["class_id"])

                for ppe in PPE_CLASSES:
                    has_must_ppe = ppe in person_ppe
                    if not has_must_ppe:
                        status_frame = 0
                        break   
        
        if status_frame is not None:
            frames_status[counter] = status_frame

            status_frame_1_qty = frames_status.count(1)
            status_frame_0_qty = frames_status.count(0)
        
            if (status_frame_1_qty + status_frame_0_qty) == QTY:
                if status_frame_1_qty >= status_frame_0_qty:
                    print("PPE OK")  
                    # GPIO.output(green_led, True)
                    # GPIO.output(red_led, False)
                else:
                    print("PPE NOK")  
                    # GPIO.output(green_led, False)
                    # GPIO.output(red_led, True)

            counter += 1
                
            if counter == QTY:
                counter = 0
                        
        # Mostra a imagem com as detecções
        # RETIRAR PARA EMBARCAR
        cv2.imshow("Detecção de EPIs", annotated_frame)
        # Pressione Q para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Programa encerrado")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # GPIO.output(green_led, False)
    # GPIO.output(red_led, False)
    # GPIO.cleanup()