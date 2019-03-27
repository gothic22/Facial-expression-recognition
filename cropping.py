import os
import cv2
 
# 读取图像，然后将人脸识别并裁剪出来, 参考https://blog.csdn.net/wangkun1340378/article/details/72457975
def clip_image(input_dir, output_dir,size1):
    images = os.listdir(input_dir)
 
    for imagename in images:
        imagepath = os.path.join(input_dir , imagename)
        img = cv2.imread(imagepath)
 
        path = r"D:\Users\william\Anaconda3\envs\tensorflow-gpu\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml"
 
        hc = cv2.CascadeClassifier(path)
 
        faces = hc.detectMultiScale(img)
        i = 1
        image_save_name = output_dir + imagename
        for face in faces:
            imgROI = img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            imgROI = cv2.resize(imgROI, (size1, size1), interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_save_name, imgROI)
            i = i + 1
        print("the {}th image has been processed".format(i))
 
 
def main():
    input_dir = "G:/dataset/125/sad1/"
    output_dir = "G:/dataset/125/Sad/"
    size1=125
    if not os.path.exists(output_dir ):
        os.makedirs(output_dir )
 
    clip_image(input_dir, output_dir,size1)
 
 
if __name__ == '__main__':
    main()
