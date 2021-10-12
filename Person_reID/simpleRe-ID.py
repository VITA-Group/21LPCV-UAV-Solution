# -*- coding: utf-8 -*-
"""
@author: Hao Yu Miao
"""
import torch
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)
    return model

def extract_feature(model,img):
    #print(img.shape)
    #print(img)
    #print(Variable(img))
    feature = model(Variable(img))
    # norm feature
    fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
    feature = feature.div(fnorm.expand_as(feature))
    return feature

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,64), interpolation=3),
    #transforms.ToTensor()
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_transforms2 = transforms.Compose([
    transforms.ToTensor()
])

cpu_device = torch.device("cpu:0")
#model = load_torchscript_model(model_filepath='./model/ResNet18_Qt/best_model_qnn.torchscript', device=cpu_device)
#print(model)
#model = model.eval()
img = cv2.imread("p100000060.jpg")

print(img)


img1 = data_transforms(img)
#b, g, r = img1.split()
#img1 = Image.merge("RGB", (r, g, b))
#img1.save("0001_c1_s1_001051_00_pil.jpg")
img1 = np.asarray(img1)
print(np.min(img1))
print(np.max(img1))
print(img1)

'''img_test = Image.fromarray(img1)
print("img_1")
print(np.asarray(img_test).transpose((2, 1, 0)))

img1 = img1.transpose((2, 1, 0))
s = img1.shape



print(img1.shape)
print(img.shape)

#img2 = cv2.resize(img, (128,64), interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img, (64,128), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("cv2_img.jpg", img2)
img_test = Image.fromarray(img2)
print("img_2")
print(np.asarray(img_test).transpose((2,0,1)))'''



'''img_test.save('img2.jpg')

img2 = np.asarray(img2)

#print(img2.shape)
img2 = img2.transpose((2,0,1))
#print(img1.shape)
#print(img2.shape)
#print(np.subtract(img1, img2)[0][0][0])

#print(img1)
#print(img2)

new = np.subtract(img1.astype('int'), img2.astype('int'))
print(np.mean(np.absolute(new)))'''


#print('img1')
#print(img1)
#print('img2')
#print(img2)


#print(img1 - img2)
#print(max(img1 - img2))


#with torch.no_grad():
    #feature = extract_feature(model, img)


#print(feature)'''