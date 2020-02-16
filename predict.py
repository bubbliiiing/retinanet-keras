from keras.layers import Input
from retinanet import Retinanet
from PIL import Image

retinanet = Retinanet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = retinanet.detect_image(image)
        r_image.show()
retinanet.close_session()
    