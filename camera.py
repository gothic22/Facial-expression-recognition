import cv2


cap = cv2.VideoCapture(0)
ret=cap.set(3,360)  #width
ret=cap.set(4,270)   #height
while(1):
    # get a frame
    ret, img1 = cap.read()
    # show a frame
    cv2.imshow("capture", img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转为灰度图片
        #tempimg = cv2.resize(img,(80,60),cv2.INTER_LINEAR)  #(width,height)
        #cv2.imwrite("C:/Users/william/Python Files/photo/face2.jpeg", tempimg)
        #cropimg = tempimg[10:58, 16:64]
        cv2.imwrite(r"C:\Users\william\Python Files\Facial-Expression-Recognition.Pytorch-master\images\face9.jpeg", img1)
        break
cap.release()
cv2.destroyAllWindows()
