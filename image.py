import cv2
class Image:
    def detect(image,type="DICT_5X5_100"):
        #resize the image
       
        # load the ArUCo dictionary, grab the ArUCo parameters, and detect
        # the markers
        #print("[INFO] detecting '{}' tags...".format(type))
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
            parameters=arucoParams)
        #print(len(corners))
    
        if len(corners) > 0:
            print(len(corners))
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                #=====================
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
        
                return topRight, bottomRight, bottomLeft,topLeft
            #return cX,cY #only last object is returned in this case. if there are multiple objects we need to create a list 
        return -1, -1, -1, -1