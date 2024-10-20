import cv2 

def camera_position(index=0):
    access_position = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else: 
            print(f"可用的攝像頭編號: {index}")
            break    
    cap.release()
    index += 1


def check_camera(index):
    
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print("未找到攝影鏡頭")
    else:
        print("鏡頭已開啟")
    return cap



    
if __name__=="__main__":
    cameras = camera_position()
