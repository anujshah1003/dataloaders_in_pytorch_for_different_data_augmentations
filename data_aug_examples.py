# -*- coding: utf-8 -*-
import os
import torchvision.transforms as transforms
from PIL import Image

root_path=r'D:\youtube\2021\data_aug_pytorch\dogs-vs-cats\train'
img_path=os.path.join(root_path,'dog.17.jpg')

img=Image.open(img_path)
img

t1=transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
t2=transforms.RandomRotation(degrees=(0, 180))
t3=transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))

t4=transforms.RandomCrop(size=(200, 200))

t5=transforms.FiveCrop(224)

t_c=transforms.Compose([t4,t2])

img_1=t1(img)
img_2=t2(img)
img_3=t3(img)
img_4=t4(img)
img_5=t5(img)
img_c=t_c(img)

img_1
img_2
img_3
img_4

for img_ in img_5:
    img_
