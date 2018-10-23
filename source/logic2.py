
def mergeTwoImage(img1, img2, isDebug=False):
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    
    if len(img1)<=1:
        return img2
    if len(img2)<=1:
        return img1

    # img1 = cv2.imread('C:\\Learn\Phyton\Panorama\\source\\test\\2\\1_1.jpg',1)
    # img2 = cv2.imread('C:\\Learn\Phyton\Panorama\\source\\test\\2\\1_2.jpg',1)
    #img1=img1[:,:,::-1] #BGR>>RGB
    #img2=img2[:,:,::-1] #BGR>>RGB
    # Initiate ORB detector


    # find the keypoints and descriptors with ORB
    #orb = cv2.ORB_create()
    #kp1, des1 = orb.detectAndCompute(img1,None)
    #kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    #matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    #matches = sorted(matches, key = lambda x:x.distance)

    kp1, des1 = detectAndDescribe(img1)
    kp2, des2 = detectAndDescribe(img2)

    # match features between the two images
    M = matchKeypoints(kp1, kp2, des1, des2, 0.75, 4.0)


    (matches, H, status) = M
    hX=int(H[0,2])*-1
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    
    result[0:img2.shape[0], hX:hX+img2.shape[1]] = img2
    
    

    return result

    if isDebug:
        # Draw first 10 matches.
        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50] ,None, flags=2)
        vis = drawMatches(img1, img2, kp1, kp2, matches,status)
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

    # # detect keypoints in the image
    # detector = cv2.FeatureDetector_create("SIFT")
    # kps = detector.detect(gray)
    # # extract features from the image
    # extractor = cv2.DescriptorExtractor_create("SIFT")
    # (kps, features) = extractor.compute(gray, kps)
    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])
    # return a tuple of keypoints and features
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
    import numpy as np
    import cv2
    
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
    if len(matches) > 4:
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