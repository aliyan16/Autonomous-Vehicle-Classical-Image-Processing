import numpy as np
import cv2 as cv



def conv(img,filter):
    img = img.astype(np.int16)
    rows,cols=img.shape
    value=0
    for i in range(rows):
        for j in range(cols):
            # value=value+((img[i,j]*filter[i,j]))
            value=(value)+((img[i,j])*(filter[i][j]))
    return value


def magnitude(img1,img2):
    rows,cols=img1.shape
    resultant=img1.copy()
    value=0
    # for i in range(rows):
    #     for j in range(cols):
    #         value=np.sqrt((img1[i,j]**2)+(img2[i,j]**2))
    #         resultant[i,j]=value
    resultant=np.sqrt((img1**2)+(img2**2))

    resultant=norm(resultant)
    return resultant

def Phase(img1,img2):
    rows,cols=img1.shape
    resultant=img1.copy()
    value=0
    # for i in range(rows):
    #     for j in range(cols):
    #         value=np.arctan2(img1[i,j],img2[i,j])
    #         resultant[i,j]=value
    resultant = np.arctan2(img2, img1) * (180.0 / np.pi)
    resultant=norm(resultant)
    return resultant

def norm(img):
    imgm = (img / np.max(img) * 255).astype(np.uint8)
    return imgm



def sobel(img,sobelx,sobely):
    rows,cols=img.shape
    magImg=img.copy()
    phaseImg=img.copy()
    # frows,fcols=sobelx.shape

    for i in range(rows-2):
        for j in range(cols-2):
            window=img[i:i+3,j:j+3]
            value1=conv(window,sobelx)
            value2=conv(window,sobely)
            magImg[i,j]=value1
            phaseImg[i,j]=value2
    return magImg,phaseImg



def cannyedge():
    sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobely = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    gx,gy=sobel(img,sobelx,sobely)
    SobelMag=magnitude(gx,gy)
    SobelPhase= Phase(gx,gy)












if __name__=='__main__':
    img=cv.imread()
