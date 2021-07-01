from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from gatys import Gatys


img = Image.open('img.jpg')
img_style = Image.open('img_style.jpg')
img = asarray(img)
img_style = asarray(img_style)

image_result = Gatys().transfer(img, img_style) # feed me 2 images and go brrr
plt.imshow(image_result)
image = Image.fromarray(image_result.astype('uint8'), 'RGB')
image.save("result.jpg")
