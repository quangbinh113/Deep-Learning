from bsm import BSMx
from lcv import LCVx
from preprocessing import get_highpass, get_wiener, old_wiener
import numpy as np
import cv2 as cv


def FSM(img:np.ndarray,func = 'heavyside', adaptive_param = 30, toh = 100, t_lcv = 0):
    res = BSMx(img, func)*LCVx(img,toh, thresh = t_lcv)

    res_mean = np.mean(res)
    res_std = (np.mean(res**2)-res_mean**2)**0.5
    return res,res_mean + adaptive_param*res_std

if __name__ == "__main__":
    PATH = '/home/irst/Manhtb2/IRST/IRST_Data/22-09-22/22-09-22-19-36-53.mp4'
    PATH = '/home/manhtb2/Desktop/manhtb2/Sim_Video/22-08-02/Sample_12.avi'
    PATH = 'Sample_13_gen.avi'
    vid = cv.VideoCapture(PATH)
    frame_no = 0
    while vid.isOpened():
        ok, frame = vid.read()
        if not ok:
            break
        framez = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        framez = get_highpass(get_wiener(framez),mode = 2)
        #frame = get_highpass(frame,mode = 2)
        framez = np.float32(framez)

        res,th= FSM(framez)
        peak = np.max(res)
        #dummy =  np.uint8((res - res.min()) * (1/(res.max() - res.min()) * 255))
        dummy = np.uint8(np.where(res>th,255,0))
        dummy = np.uint8(np.where(res>0.3*np.max(res),255,0))
        contour, hiearachy = cv.findContours (dummy,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


        for c in contour:
            if len(c)>=3:
                x,y,w,h = cv.boundingRect(c)
                cv.rectangle(frame, (x-3,y-3),(x+w+3,y+h+3),(0,0,255), 1)

        cv.imshow('sur',dummy)
        cv.imshow('frame', frame)
        #exit()
        if cv.waitKey(1) == ord('x'):
            vid.release()

        frame_no+=1


# if __name__ == "__main__":
#     PATH = '/home/irst/Manhtb2/IRST/IRST_Data/22-09-22'
#     frame = cv.imread(PATH)
    
#     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     cv.imshow('ori',frame)
#     frame = get_highpass(old_wiener(frame),mode = 2)
#     cv.imshow('prep',frame)
#     frame = np.float32(frame)

#     res, thresh = FSM(frame,'ReLU',50)
#     show_img = np.uint8(np.where(res>thresh,255,0))
#     cv.imshow('res', show_img)
#     if cv.waitKey(0) == ord('x'):
#         exit()