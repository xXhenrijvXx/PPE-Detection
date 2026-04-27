from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo26n.pt")

# Train the model on Construction-PPE dataset
model.train(data="data.yaml", epochs=100, imgsz=640)

# RESULTADO

# 100 epochs completed in 5.657 hours.
# Optimizer stripped from C:\Users\a511009\Desktop\Projeto\PPE-Detection\runs\detect\train\weights\last.pt, 5.4MB
# Optimizer stripped from C:\Users\a511009\Desktop\Projeto\PPE-Detection\runs\detect\train\weights\best.pt, 5.4MB

# Validating C:\Users\a511009\Desktop\Projeto\PPE-Detection\runs\detect\train\weights\best.pt...
# Ultralytics 8.4.41  Python-3.14.2 torch-2.11.0+cpu CPU (12th Gen Intel Core i7-12850HX)
# YOLO26n summary (fused): 122 layers, 2,376,981 parameters, 0 gradients, 5.2 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 1.2s/it 6.0s
#                    all        143       1172      0.626      0.521      0.544      0.267
#                 helmet        107        201      0.723      0.811      0.774      0.413
#                 gloves         68        136      0.782      0.684      0.727      0.365
#                   vest        109        171      0.803      0.761      0.804      0.495
#                  boots         64        151      0.717      0.762      0.771      0.421
#                goggles         44         47       0.69      0.663      0.663      0.319
#                   none         43         81      0.609      0.577      0.542      0.193
#                 Person        139        239      0.837      0.878      0.892      0.496
#              no_helmet         27         45      0.341      0.196      0.253     0.0823
#              no_goggle         25         41          0          0      0.092     0.0289
#              no_gloves         23         56      0.554      0.155      0.185      0.066
#               no_boots          2          4      0.831       0.25      0.276     0.0577
# Speed: 0.6ms preprocess, 34.4ms inference, 0.0ms loss, 0.0ms postprocess per image
# Results saved to C:\Users\a511009\Desktop\Projeto\PPE-Detection\runs\detect\train