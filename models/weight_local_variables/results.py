from fsm_thresh import FSM
from preprocessing import get_highpass, get_wiener, old_wiener
import numpy as np
import cv2 as cv

# data = dict()

# with open('/home/manhtb2/Desktop/manhtb2/Sim_Video/22-08-02/gt/Sample_66.txt') as f:
#     for fline in f.readlines():
#         flinez = fline.splitlines()[0]
#         check = [int(i) for i in flinez.split(',')]
#         data[check[0]] = check[1:]


PATH = 'sample/Sample_16.avi'
vid = cv.VideoCapture(PATH)

frames = vid.get(cv.CAP_PROP_FRAME_COUNT)
# fps = vid.get(cv.CAP_PROP_FPS)
# vid_time = frames/fps

frame_no = 2
detected = 0
detected_small = 0
err = 0
while vid.isOpened():
    ok, frame = vid.read()
    if not ok:
        break
    
    framez = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    alarm = 0

    framez = get_highpass(get_wiener(framez),mode = 2)
    #framez = get_highpass(frame,mode = 2)
    framez = framez.astype('float64')
    res,thresh = FSM(framez)
    sur = np.uint8(np.where(res>0.3*np.max(res),255,0))

    contour, hiearachy = cv.findContours (sur,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    
    for c in contour:
        if len(c)>=3:
            alarm += 1
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(frame, (x-3,y-3),(x+w+3,y+h+3),(0,0,255), 1)
    
    # coor_x, coor_y = data[frame_no][:2]

    # detect_val = np.sum(sur[max(0,coor_y-6):coor_y+6,coor_x-6:coor_x+6])
    # if detect_val>= 255*3:
    #     err += alarm - 1
    #     detected += 1
    #     detected_small += 1
    # else:
    #     if detect_val>10:
    #         detected_small += 1
    #     err += alarm
        #exit()
    
    cv.imshow('frame',frame)
    cv.imshow('sur',sur)

    if cv.waitKey(1) == ord('x'):
        vid.release()
    if frame_no%20 == 0:
        print('%i frames processed' %frame_no)
    frame_no += 1

# print(frames)
# print (f'Ratio: {detected/(frames)*100}%')
# print (f'Ratio small: {detected_small/(frames)*100}%')
# print (f'False alarms per frame: {err/frames}')
