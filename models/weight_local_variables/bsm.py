import numpy as np
import cv2 as cv
from scipy.signal import convolve2d
from numpy import ndarray
from preprocessing import get_highpass, old_wiener, get_wiener


def kernelgen (type: str, number, K = 3):
    if type == 'OB':
        kernel = np.ones((5*K, 5*K))
        kernel[K:4*K,K:4*K] = np.zeros((3*K,3*K))
        kernel /= (16*K**2)
    elif type == 'PB':
        kernel = np.zeros((5*K, 5*K))
        if number == 0:
            kernel[K:K*4,K:K*4] = np.ones((3*K,3*K))
            kernel[K*2:K*3,K*2:K*3] = np.zeros((K,K))
            kernel /= 8*K**2
        elif number in range(1,9):
            if number in range(1,4):
                kernel [K:K*2, number*K:K*(number+1)] = np.ones((K,K))              
            elif number in range(5,8):
                kernel [K*3:K*4, K*(8-number):K*(9-number)] = np.ones((K,K))
            elif number == 8:
                kernel [K*2:K*3, K:K*2] = np.ones((K,K))
            else:
                kernel [K*2:K*3, K*3:K*4] = np.ones((K,K))
            kernel /= K**2
        else:
            raise Exception('2nd argument must be in range[0,4]') 
    elif type == 'I':
        kernel = np.zeros((5*K,5*K))
        kernel[K*2:K*3,K*2:K*3] = np.ones((K,K))
        kernel /= K**2
    else:
        raise Exception('1st argument of kernelgen() must be "OB", "PB" or "I"') 
    return kernel

def heavyside(mat: np.ndarray):
    return np.where(mat>0,1,0)

def ReLU(mat: np.ndarray):
    return np.int64(np.where(mat>0,mat,0))

# Calculate mean
def meanI(img: np.ndarray):
    sliding_window_I = kernelgen('I',0)
    return cv.filter2D(img,-1,sliding_window_I,borderType= cv.BORDER_REFLECT)

def meanOB(img: np.ndarray):
    sliding_window_OB = kernelgen('OB',0)
    return cv.filter2D(img,-1,sliding_window_OB,borderType= cv.BORDER_REFLECT)

def meanPB(img: np.ndarray, number: int):
    sliding_window_PB = kernelgen('PB',number)
    return cv.filter2D(img,-1,sliding_window_PB,borderType= cv.BORDER_REFLECT)

def meanIPB(img: np.ndarray, K = 3):
    kernel = np.zeros((5*K, 5*K))
    kernel[K:4*K,K:4*K] = np.ones((3*K,3*K))
    kernel /= (9*K**2)
    return cv.filter2D(img,-1,kernel,borderType=cv.BORDER_REFLECT)


# Calculate BC
#@njit(fastmath= True, parallel = True)
def BC(img: np.ndarray, func = 'heavyside'):
    module1 = np.int32(meanI(img)) - np.int32(meanPB(img,0))
    module2 = np.int32(meanI(img)) - np.int32(meanOB(img))
    if func == 'heavyside':
        return heavyside(module1 * module2) * heavyside(module1 + module2)
    if func == 'ReLU':
        return ReLU(module1 * module2) * ReLU(module1 + module2)


# Calculate DI
def DI(img: np.ndarray):
    tar = np.float32(meanI(img))
    store_img = []
    for i in range(1,5):
        dir1 = np.float32(meanPB(img, i))
        dir2 = np.float32(meanPB(img, i+4))
        store_img.append((tar-dir1)*(tar-dir2))
    return np.min(store_img,axis = 0)

def BSMx(img: np.ndarray,func = 'heavyside'):
    res = BC(img,func)*DI(img)
    #return res
    return res


if __name__ == "__main__":
    # PATH = '/home/manhtb2/Desktop/manhtb2/1x2_Sim/22-08-03 (copy)/5targets.avi'
    # vid = cv.VideoCapture(PATH)
  
    # while vid.isOpened():
    #     ok, frame = vid.read()
    #     if not ok:
    #         break
    #     frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #     cv.imshow('frame',frame)

    #     x = BC(frame)

    #     cv.imshow('bc',np.uint8(np.where(frame*x>140,frame*x,0)))
    #     if cv.waitKey(1) == ord('x'):
    #         vid.release()




    PATH = '/home/manhtb2/Desktop/manhtb2/1x2_Sim/22-08-03 (copy)/5targets.avi'
    PATH = '/home/manhtb2/Desktop/manhtb2/Sim_Video/22-08-02/Sample_1.avi'
    vid = cv.VideoCapture(PATH)
    frame_no = 0
    while vid.isOpened():
        ok, frame = vid.read()
        frame_no += 1
        if not ok:
            break
        # if frame_no == 10:
        framed = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
       
        framez = get_highpass(get_wiener(framed),2)
        framez = np.float32(framez)


        dummy = BSMx(framez)
        th = np.max(dummy)
        sur = np.uint8(np.where(dummy>0.35*th,255,0))
        contour, hiearachy = cv.findContours (sur,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


        for c in contour:
            if len(c)>=3:
                x,y,w,h = cv.boundingRect(c)
                cv.rectangle(frame, (x-3,y-3),(x+w+3,y+h+3),(0,0,255), 1)

        cv.imshow('frame',np.uint8(frame))
        cv.imshow('sur',sur)
        #exit()

        if cv.waitKey(1) == ord('x'):
            vid.release()



    # PATH = '/home/irst/Downloads/data11/19.bmp'
    # img = cv.imread(PATH)
    # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # cv.imshow('original',img)
    # frame = get_highpass(old_wiener(img),2)
    # cv.imshow('frame', frame)

    # scene = BSMx(frame,'ReLU')
    # dummy =  ((scene - scene .min()) * (1/(scene .max() - scene.min()) * 255)).astype('uint8')
    # # maxv = np.max(scene)
    # cv.imshow('sur',dummy)
    # # DI_unscaled = DI(frame)
    # # dummy =  ((DI_unscaled - DI_unscaled .min()) * (1/(DI_unscaled .max() - DI_unscaled.min()) * 255)).astype('uint8')
    # # cv.imshow('DI',np.where(dummy>100,dummy,0))

    # if cv.waitKey(0000) == ord('x'):
    #     exit()