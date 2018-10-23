
#merge two images
def mergeTwoImage(image1, image2):
    import numpy as np
    import constans as c

    if len(image1)<=1:
        return image2
    if len(image2)<=1:
        return image1

    maxD1=len(image1)
    maxD2=len(image1[0])+len(image2[0])
    totalImage=np.zeros((maxD1,maxD2,3), c.IMAGE_ARRAY_TYPE)

    print("image1: ",len(image1),len(image1[0]))
    print("image2: ",len(image2),len(image2[0]))
    print("totalImage: ",len(totalImage),len(totalImage[0]))

    correctOffset=0
    trying=0
    while trying<c.MAXTRYING and correctOffset==0:
        sampling=getSampleFromImage(image2)
        currentD1=0
        correctCounter=0
        while currentD1<len(image1):
            currentD2=0
            while currentD2<len(image1[currentD1]):
                totalImage[currentD1,currentD2]=image1[currentD1,currentD2]
                if checkRGBs(image1[currentD1,currentD2],sampling[correctCounter]):
                    correctCounter+=1
                    if correctCounter==c.SAMPLING_SIZE:
                        a=1
                        while a<c.SAMPLING_SIZE+1:
                            printRGB(image1[currentD1,currentD2-a+1])
                            printRGB(sampling[correctCounter-a])
                            a+=1
                        correctOffset=currentD2
                        correctCounter=0
                        print("HIT!")
                else:
                    correctCounter=0
                currentD2+=1
            currentD1+=1
        trying+=1
    offsetD2=currentD2-1
    if correctOffset!=0:
        offsetD2=correctOffset
    currentD1=0
    while currentD1<len(image1):
        currentD2=0
        while currentD2<len(image2[currentD1]):
            totalImage[currentD1,currentD2+offsetD2]=image2[currentD1,currentD2]
            currentD2+=1
        currentD1+=1

    return totalImage

def checkRGBs(rgb1,rgb2):
    #return getIfromRGB(rgb1)==getIfromRGB(rgb2)
    maxTolarence=3
    return abs(rgb1[0]-rgb2[0])<maxTolarence and abs(rgb1[1]-rgb2[1])<maxTolarence and abs(rgb1[2]-rgb2[2])<maxTolarence

def getIfromRGB(rgb):
    return rgb[0]*256**2 + rgb[1]*256 +rgb[2]

def printRGB(rgb):
    print(rgb[0],"-",rgb[1],"-",rgb[2])

def getSampleFromImage(image):
    import constans as c
    import random as r
    import numpy as np

    sampling=np.zeros((c.SAMPLING_SIZE,3), c.IMAGE_ARRAY_TYPE)
    samplingCounter=0
    samplingRow=int(round(len(image)/2))+r.randint(-30, 30)
    print("samplingRow: ",samplingRow)
    while samplingCounter<c.SAMPLING_SIZE:
        sampling[samplingCounter]=image[samplingRow,samplingCounter]
        samplingCounter+=1
    return sampling