import numpy as np
import cv2 as cv
from bsm import meanOB, meanIPB, BSMx
from preprocessing import get_highpass, old_wiener, get_wiener

def stdIPB(img:np.ndarray):
    meanT = np.int32(meanIPB(img))
    meanTsquared = np.int32(meanIPB(img**2))
    pre_s = (meanTsquared-meanT**2)
    return np.sqrt(pre_s*9/8)
    return (np.where(pre_s>0,pre_s,0)*9/8)**0.5

def stdOB(img:np.ndarray):
    meanO = np.int32(meanOB(img))
    meanOsquared = np.int32(meanOB(img**2))
    pre_s = ((meanOsquared-meanO**2))
    return (pre_s*16/15)**0.5
    return (np.where(pre_s>0,pre_s,0)*16/15)**0.5

def LCVx(img:np.ndarray,toh = 100,thresh = 10):
    res = (2*stdIPB(img)-stdOB(img))/(meanOB(img)+toh)
    return res
    dummy =  np.uint8((res - res.min()) * (1/(res.max() - res.min()) * 255))
    dummy = np.where(dummy>100, dummy, 0)
    #dummy = np.uint8(np.where(dummy>200,255,0))
    return dummy
    #return np.uint8(np.where(res>0.4*np.max(res),255,0))
    return np.uint8(np.where(res>thresh,255,0))

if __name__ == "__main__":
    PATH = '/home/manhtb2/Desktop/manhtb2/Sim_Video/22-08-02/Sample_12.avi'
    vid = cv.VideoCapture(PATH)
    frame_no = 0
    while vid.isOpened():
        ok, frame = vid.read()
        frame_no += 1
        if not ok:
            break
        if frame_no%1 == 0:

            cv.imshow('frame',np.uint8(frame))
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            frame = get_highpass(get_wiener(frame),2)
            frame = np.float32(frame)


            cv.imshow('sur',np.uint8(np.where(LCVx(frame)>100,255,0)));cv.imshow('sur',LCVx(frame))
            #exit()
        if cv.waitKey(1) == ord('x'):
            vid.release()


    # PATH = '/home/irst/Downloads/data11/29.bmp'
    # img = cv.imread(PATH)
    # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # cv.imshow('original',img)
    # frame = get_highpass(old_wiener(img),2)
    # cv.imshow('frame', frame)
    # frame = np.float32(frame)

    # dummy = LCVx(frame)
    # print(np.min(dummy),np.max(dummy))
    # dummy = np.where(dummy>0.45*np.max(dummy),255, 0)
    # # rint(np.min(dummy),np.max(dummy))
    # #dummy =  ((dummy - dummy.min()) * (1/(dummy.max() - dummy.min()) * 255)).astype('uint8')
    # cv.imshow('lcv',np.uint8(dummy))

    # if cv.waitKey(0000) == ord('x'):
    #     exit()