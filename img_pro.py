import csv,os,cv2
import os.path
import glob
from PIL import Image
import numpy as np


def convert_img_to_csv(i,img_dir,size1,label):
    #设置需要保存的csv路径
    with open("G:/Facial-Expression-Recognition.Pytorch-master/data/"+str(size1)+".csv","a",newline="") as f:
        writer = csv.writer(f)
        #设置csv文件的列名
        if i==0:            
            column_name = []
            column_name.append('emotion')
            column_name.extend(["pixel%d"%t for t in range(size1*size1)])
            column_name.append('usage')
            #将列名写入到csv文件中
            writer.writerow(column_name)
        else:
            pass
        img_temp_dir = os.path.join(img_dir)
        #获取该目录下所有的文件
        img_list = os.listdir(img_temp_dir)
        #遍历所有的文件名称
        i_num=0
        f_num=len(img_list)
        for img_name in img_list:
            #判断文件是否为目录,如果为目录则不处理
            if not os.path.isdir(img_name):
                #获取图片的路径
                img_path = os.path.join(img_temp_dir,img_name)
                #因为图片是黑白的，所以以灰色读取图片
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

                #图片标签
                row_data = [label]
                #获取图片的像素
                row_data.extend(img.flatten())
                if i_num < int(f_num*0.7):
                    row_data.append('Training')
                elif i_num<int(f_num*0.85):
                    row_data.append('PublicTest')
                else:
                    row_data.append('PrivateTest')                    
                #将图片数据写入到csv文件中
                writer.writerow(row_data)
                i_num+=1


def to_jpg():
    i=0
    for file in glob.glob(r'G:\dataset\image\2_gray\*.tif'): 
        im=Image.open(file)
        im.save('G:/dataset/image/CAS_EXP/'+str(i)+'.jpg')
        i+=1
    return

def re_size(in_path,out_path,size1):    
    for file in glob.glob(in_path): 
        i=np.random.randint(10000)
        img=cv2.imread(file)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(size1,size1),cv2.INTER_LINEAR)
        cv2.imwrite(out_path+str(i)+".jpg",img)
    return
#
#in_path="G:/dataset/125/Sad/*.jpg"
#out_path="G:/dataset/125/4_CAS/"
#re_size(in_path,out_path,125)        

