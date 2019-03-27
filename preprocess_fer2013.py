# create data and label for FER2013
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv
import os
import numpy as np
import h5py
import img_pro
from models.vgg import size1,num_label

def get_csv(in_path,out_path,size1,num_label):
    for label in range(num_label):
        img_pro.re_size(in_path+str(label)+"_MIX/*.jpg",out_path+str(size1)+"/"+str(label)+"_MIX/",size1)
        img_pro.convert_img_to_csv(label,out_path+str(size1)+"/"+str(label)+"_MIX/",size1,label)
    return



if not os.path.exists("G:/Facial-Expression-Recognition.Pytorch-master/data/"+str(size1)+".csv"):
    in_path="G:/dataset/125/"
    out_path="G:/dataset/"
    get_csv(in_path,out_path,size1,num_label)

file = "G:/Facial-Expression-Recognition.Pytorch-master/data/"+str(size1)+".csv"

# Creat the list to store the data and label information
Training_x = []
Training_y = []
PublicTest_x = []
PublicTest_y = []
PrivateTest_x = []
PrivateTest_y = []

datapath = os.path.join('data','data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

with open(file,'r') as csvin:
    data=csv.reader(csvin)
    for row in data:
        l1=len(Training_x)
        l2=len(PublicTest_x)
        l3=len(PrivateTest_x)
        if row[-1] == 'Training':
            temp_list = []
            pixels=row[1:-1]
            for pixel in pixels:
                temp_list.append(int(pixel))
            t1=np.random.randint(l1+1)
            I = np.asarray(temp_list)
            Training_y.insert(t1,int(row[0]))
            Training_x.insert(t1,I.tolist())

        if row[-1] == "PublicTest" :
            temp_list = []
            pixels=row[1:-1]
            for pixel in pixels:
                temp_list.append(int(pixel))
            t2=np.random.randint(l2+1)
            I = np.asarray(temp_list)
            PublicTest_y.insert(t2,int(row[0]))
            PublicTest_x.insert(t2,I.tolist())

        if row[-1] == 'PrivateTest':
            temp_list = []
            pixels=row[1:-1]
            for pixel in pixels:
                temp_list.append(int(pixel))
            t3=np.random.randint(l3+1)
            I = np.asarray(temp_list)
            PrivateTest_y.insert(t3,int(row[0]))
            PrivateTest_x.insert(t3,I.tolist())

num_train=np.shape(Training_x)[0]
num_private=np.shape(PrivateTest_x)[0]
num_public=np.shape(PublicTest_x)[0]
print(np.shape(Training_x))
print(np.shape(PublicTest_x))
print(np.shape(PrivateTest_x))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("Training_pixel", dtype = 'uint8', data=Training_x)
datafile.create_dataset("Training_label", dtype = 'int64', data=Training_y)
datafile.create_dataset("PublicTest_pixel", dtype = 'uint8', data=PublicTest_x)
datafile.create_dataset("PublicTest_label", dtype = 'int64', data=PublicTest_y)
datafile.create_dataset("PrivateTest_pixel", dtype = 'uint8', data=PrivateTest_x)
datafile.create_dataset("PrivateTest_label", dtype = 'int64', data=PrivateTest_y)
datafile.close()

print("Save data finish!!!")
