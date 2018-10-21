
#merge two images
def mergeTwoImage(image1, image2):
    import numpy as np
    
    if len(image1)<=1:
        return image2
    if len(image2)<=1:
        return image1

    
  
    maxD1=len(image1)
    maxD2=len(image1[0])+len(image2[0])
    totalImage=np.zeros((maxD1,maxD2,3), 'uint8')


    sampling=np.zeros((10,3), 'uint8')
    samplingCounter=0
    samplingCenter=int(round(maxD1/2))-5
    while samplingCounter<10:
        sampling[samplingCounter]=image1[0,samplingCenter+samplingCounter]
        samplingCounter+=1
    
    print("image1: ",len(image1),len(image1[0]))
    print("image2: ",len(image2),len(image2[0]))
    print("totalImage: ",len(totalImage),len(totalImage[0]))


    currentD1=0
    while currentD1<len(image1):
        currentD2=0
        while currentD2<len(image1[currentD1]):
            totalImage[currentD1,currentD2]=image1[currentD1,currentD2]
            currentD2+=1
        currentD1+=1
    offsetD2=currentD2-1    # ez a középvonal is!
    currentD1=0
    while currentD1<len(image2):
        currentD2=0
        while currentD2<len(image2[currentD1]):
            totalImage[currentD1,currentD2+offsetD2]=image2[currentD1,currentD2]
            currentD2+=1
        currentD1+=1

    return totalImage