import argparse
import utils as ut
import cv2
import time
import numpy as np
from PIL import Image
def main(model,style_id,style_id2,class_id):
    #CAMERA_ID="rtmp://140.116.56.6:1935/live"
    CAMERA_ID=0
    WIDTH = 640
    HEIGHT = 480
    
    cv2.namedWindow("Realtime Style-Transfer")

    # 開啟攝像頭，並設置長寬
    vc = cv2.VideoCapture(CAMERA_ID)
    vc.set(cv2.CAP_PROP_FPS, 15)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    print("Loading model")
    seg_model = ut.create_seg_model(model)
    style_model = ut.create_style_model(style_id)
    style_model_bg=ut.create_style_model(style_id2)
    print("Load model succeccfully!")
    # 檢查攝像頭是否成功打開
    if vc.isOpened():
        # 讀取攝像頭畫面
        rval, frame = vc.read()
    else:
        rval = False
        
    while rval:
        # 顯示攝像頭畫面
        cv2.imshow("Realtime Style-Transfer", frame)

        # 讀取攝像頭畫面
        rval, frame = vc.read()
        dup=frame    
        # 記錄開始時間
        start = time.time()

        # 對畫面進行風格轉換並調整大小
        frame = cv2.resize(CBStyling(model,frame,seg_model,style_model,style_model_bg,style_id,class_id), (0, 0), fx=1.0, fy=1.0)

        # 輸出風格轉換所花費的時間
        print(time.time() - start, 'sec')
            
        # 等待鍵盤輸入，每隔 1 毫秒檢查一次
        key = cv2.waitKey(1)
        fps = vc.get(cv2.CAP_PROP_FPS)
        print("Video capture FPS:", fps)    
        # 如果按下 'q' 鍵，則退出迴圈，結束程式
        if 'q' == chr(key & 0xFF):
            break

    # 關閉攝像頭視窗 
    vc.release()

    # 關閉攝像頭視窗 
    cv2.destroyWindow("Realtime Style-Transfer")



def CBStyling(model,img_frame,seg_model,style_model,style_model_bg,style_id,class_id):

    # Create style and segmentation model
    
    # ======================================
    # Seg part
    image = ut.load_image(img_frame, model_name = model)
    # semseg = ut.get_semseg_image(seg_model, image)
    # Get image with mask showing class_id 13, which is cars by default
    print("Stylizing class %d..." % class_id)
    
    fg_image = ut.get_masked_image(seg_model, image, category=class_id, bg=0, model_name = model)
    # Get image with mask showing everything except class_id 13
    bg_image = ut.get_masked_image(seg_model, image, category=class_id, bg=1, model_name = model)
    #cv2.imshow("Seg", semseg)
    #key = cv2.waitKey(1)
    

    # ======================================
    # Style part
    image = ut.load_image_style(img_frame, scale=1.0)
    image_style1 = ut.get_styled_image(style_model, image)
    image_style2 = ut.get_styled_image(style_model_bg, image)
    # # Apply local style to fg
    fg_styled = image_style1 * (fg_image != 0)
    # # Apply local style to bg
    bg_styled = image_style2* (bg_image != 0)

    # ======================================
    # Save part
    # ut.save_image(out_fname, fg_styled + bg_image)
    # print("SAVED: %s" % out_fname)
    # if model = "ENet":
    #     mask = model
    # else:

    image=fg_styled + bg_styled
    image=image[:, :, ::-1]
    result=np.asarray(Image.fromarray(image.astype('uint8')))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument("-i", "--img_fname", type=str, default="", help="path of image to style")
    # parser.add_argument("-o", "--out_fname", type=str, default="", help="path of image to style")
    parser.add_argument("-m", "--model", type=str, default = "DABNet", help="Choose a model")
    parser.add_argument("-s1", "--style1", type=int, default=0, help="Choose a style 0-3")
    parser.add_argument("-s2", "--style2", type=int, default=0, help="Choose a style 0-3")
    parser.add_argument("-c", "--class_id", type=int, default=1, help="Choose a class_id 1-20")
    args = parser.parse_args()

    main(args.model,args.style1, args.style2,args.class_id)
