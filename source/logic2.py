
def mergeTwoImage(imgDest, imgSource, isDebug=False):
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

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
    M = matchKeypoints(kp1, kp2, des1, des2, 0.75, 4.0)

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
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # return the visualization
    return vis

def detectAndDescribe(image):
    import numpy as np
    import cv2
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kps, features = orb.detectAndCompute(gray,None)
    kps = np.float32([kp.pt for kp in kps])
    # return a tuple of keypoints and features
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
    import numpy as np
    import cv2
    import constans as c
    
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    # computing a homography requires at least 4 matches
    if len(matches) > c.MIN_MATCH_COUNT:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)

        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status)

    # otherwise, no homograpy could be computed
    return None