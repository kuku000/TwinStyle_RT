import cv2

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法打開鏡頭")
    exit()

while True:
    ret,frame=cap.read()
    if ret :
        cv2.imshow("Video",frame)
    
    if cv2.waitKey(1)& 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()