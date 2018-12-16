import numpy as np
import cv2
import constans as c

def mergeTwoImage(imgDest, imgSource, isDebug=False):
    #Ha a célkép üres, akkor a forráskép
    if len(imgDest)<=1:
        return imgSource
    #Ha a forráskép üres, akkor vissza a célkép
    if len(imgSource)<=1:
        return imgDest
   
    #kulcspontok keresése az összeillesztéshez
    kp1, des1 = detectAndDescribe(imgDest)
    kp2, des2 = detectAndDescribe(imgSource)

    # kulcspontok összevetése
    M = matchKeypoints(kp1, kp2, des1, des2)

    (matches, H, status) = M
    hX=int(H[0,2])*-1   # ez az érték, amivel el kell tolni a célképet
    result = cv2.warpPerspective(imgDest, H, (hX + imgSource.shape[1], imgDest.shape[0]))
    result[0:imgDest.shape[0], 0:imgDest.shape[1]] = imgDest
    result[0:imgSource.shape[0], hX:hX+imgSource.shape[1]] = imgSource
    #debug módban az összeillsztett kulcspontok megjelennek (10)
    if isDebug: 
        vis = drawMatches(imgDest, imgSource, kp1, kp2, matches[0:10],status)
        return vis
    else:
        return result

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    import numpy as np
    import cv2
    
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    return vis

def detectAndDescribe(image):
    # kép konvertálása szürkévé
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    # kulcspontok és funkciók detektálása
    kps, features = orb.detectAndCompute(gray,None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB,ratio=0.75, reprojThresh=4.0):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > c.MIN_MATCH_COUNT:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
        return (matches, H, status)
    return None