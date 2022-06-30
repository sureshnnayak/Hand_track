import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('./sift_trial/target2.jpeg',cv.IMREAD_GRAYSCALE)         # queryImage
img2 = cv.imread('sift_trial/shelf2.jpg',cv.IMREAD_GRAYSCALE) # trainImage
img3 = cv.imread('sift_trial/shelf2.jpg')


plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))

#time.sleep(10)

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
# plt.imshow(img3,),plt.show()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

THRESHOLD_MATCH_COUNT = 10
encircle_width = 50
scale_shelf=False
scale_percent=0
## Eliminating not good matches by defining threshold 
if len(good) > THRESHOLD_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
#     polyline_params = dict(isClosed=True,
#                        color=(255,0,0),
#                        thickness=10,
#                        lineType=cv2.LINE_AA,
#                        shift=0)
#     img2 = cv2.polylines(img2,[np.int32(dst)],True,(0, 255 ,0),4, cv2.LINE_AA)


else:
    # print("Not enough matches are found - {%d}/{%d}").format(len(good),MIN_MATCH_COUNT)
    print("Not enough matches are found")
    matchesMask = None

## Defining parameters of draw funciton for insterest points of both image

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                     singlePointColor = None,
#                     matchesMask = matchesMask, # draw only inliers
#                     flags = 2)
# plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
# matched_image = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# ## Here, we are encircling the search image on target one. 

# cv.line(matched_image, (int(dst[0,0,0] + img1.shape[1]), int(dst[0,0,1])),\
#     (int(dst[1,0,0] + img1.shape[1]), int(dst[1,0,1])), (0,0,255), encircle_width)
# cv.line(matched_image, (int(dst[1,0,0] + img1.shape[1]), int(dst[1,0,1])),\
#     (int(dst[2,0,0] + img1.shape[1]), int(dst[2,0,1])), (0,0,255), encircle_width)
# cv.line(matched_image, (int(dst[2,0,0] + img1.shape[1]), int(dst[2,0,1])),\
#     (int(dst[3,0,0] + img1.shape[1]), int(dst[3,0,1])), (0,0,255), encircle_width)
# cv.line(matched_image, (int(dst[3,0,0] + img1.shape[1]), int(dst[3,0,1])),\
#         (int(dst[0,0,0] + img1.shape[1]), int(dst[0,0,1])), (0,0,255), encircle_width)

# plt.imshow(cv.cvtColor(matched_image, cv.COLOR_BGR2RGB))
# plt.show()

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
# matched_image = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

## Here, we are encircling the search image on target one. 


#cv2.line(images, topLeft, topRight, (255, 0, 0), 2)

cv.line(img3, (int(dst[0,0,0]), int(dst[0,0,1])),\
    (int(dst[1,0,0] ), int(dst[1,0,1])), (0,0,255), encircle_width)
cv.line(img3, (int(dst[1,0,0]), int(dst[1,0,1])),\
    (int(dst[2,0,0] ), int(dst[2,0,1])), (0,0,255), encircle_width)
cv.line(img3, (int(dst[2,0,0]), int(dst[2,0,1])),\
    (int(dst[3,0,0]), int(dst[3,0,1])), (0,0,255), encircle_width)
cv.line(img3, (int(dst[3,0,0]), int(dst[3,0,1])),\
        (int(dst[0,0,0] ), int(dst[0,0,1])), (0,0,255), encircle_width)

plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.show()