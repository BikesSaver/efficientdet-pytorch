#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from efficientdet import EfficientDet
from PIL import Image

efficientdet = EfficientDet('./logs/Epoch29-Total_Loss0.7971-Val_Loss0.7432.pth')

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = efficientdet.detect_image(image)
        r_image.show()
