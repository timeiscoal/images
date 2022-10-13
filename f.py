import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = cv2.imread('Untitled.jpeg')
results = model(img)
results.save()

result = results.pandas().xyxy[0].to_numpy()
result = [item for item in result if item[6]=='person']

tmp_img = cv2.imread('Untitled.jpeg')
print(tmp_img.shape)
cropped = tmp_img[int(result[2][1]):int(result[2][3]), int(result[2][0]):int(result[2][2])]
print(cropped.shape)
cv2.imwrite('person5.png', cropped)
