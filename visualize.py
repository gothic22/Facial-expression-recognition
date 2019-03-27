"""
visualize results for test image
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

i1=np.random.randint(1000)
#cap = cv2.VideoCapture(0)
#ret=cap.set(3,120)  #width
#ret=cap.set(4,90)   #height
#while(1):
#    # get a frame
#    ret, img1 = cap.read()
#    # show a frame
#    cv2.imshow("capture", img1)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        #gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)# 转为灰度图片
#        tempimg = cv2.resize(img1,(80,60),cv2.INTER_LINEAR)  #(width,height)
#        #cv2.imwrite("C:/Users/william/Python Files/photo/face2.jpeg", tempimg)
#        cropimg = tempimg[10:58, 16:64]
#        path1=r"C:\Users\william\Desktop\Facial-Expression-Recognition.Pytorch-master\images\face"+str(i1)+".jpeg"
#        cv2.imwrite(path1, cropimg)
#        break
#cap.release()
#cv2.destroyAllWindows()



cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

path1=r"C:\Users\william\Desktop\Facial-Expression-Recognition.Pytorch-master\images\02307.jpg"
raw = io.imread(path1)
#gray = rgb2gray(raw)
#gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

img = raw[:, :, np.newaxis]

img = np.concatenate((img, img, img), axis=2)
img = Image.fromarray(img)
inputs = transform_test(img)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt', 'Unknown', 'Not a face']

net = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

ncrops, c, h, w = np.shape(inputs)

inputs = inputs.view(-1, c, h, w)
inputs = inputs.cuda()
with torch.no_grad():
    inputs = Variable(inputs)
outputs = net(inputs)

outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

score = F.softmax(outputs_avg,0)
_, predicted = torch.max(outputs_avg.data, 0)

plt.rcParams['figure.figsize'] = (13.5,5.5)
axes=plt.subplot(1, 3, 1)
plt.imshow(raw,cmap = plt.get_cmap('gray'))
plt.xlabel('Input Image', fontsize=16)
axes.set_xticks([])
axes.set_yticks([])
plt.tight_layout()


plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

plt.subplot(1, 3, 2)
ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
width = 0.4       # the width of the bars: can also be len(x) sequence
color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy','violet','gold','black']
for i in range(len(class_names)):
    plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
plt.title("Classification results ",fontsize=20)
plt.xlabel(" Expression Category ",fontsize=16)
plt.ylabel(" Classification Score ",fontsize=16)
plt.xticks(ind, class_names, rotation=45, fontsize=14)

axes=plt.subplot(1, 3, 3)
emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
plt.imshow(emojis_img)
plt.xlabel('Emoji Expression', fontsize=16)
axes.set_xticks([])
axes.set_yticks([])
plt.tight_layout()
# show emojis

#plt.show()
path2=r"C:\Users\william\Desktop\Facial-Expression-Recognition.Pytorch-master\images\results\result"+str(i1)+".jpeg"
plt.savefig(os.path.join(path2))
plt.close()

print("The Expression"+str(i1)+" is %s" %str(class_names[int(predicted.cpu().numpy())]))