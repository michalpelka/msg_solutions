import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json

DEBUG_DRAW = True

#### ground truth data from Blender ####
ground_truth_points = """
{
    "c1_uv": {
        "1": [532.6151462599752, 310.86647580643603, 1.0],
        "2": [383.00793615527954,255.4229430524797,1.0],
        "3": [301.81264237762065, 339.29727243585,1.0],
        "4": [316.12077560166944, 157.1359580558737,1.0],
        "5": [236.3578248165133, 211.03502784178835, 1.0],
        "6": [41.05770226771547, 157.13582529238067, 1.0],
        "7": [378.195637776906, 58.91817405041178, 1.0],
        "8": [180.0269024041258,152.0650246955493,1.0],
        "9": [147.07041917895486,93.94790094909477,1.0],
        "10": [526.1244994581041,32.397733391696946,1.0],
        "11": [306.3146351303051,39.59835929699141,1.0],
        "12": [257.34251876644566,97.86480150082888,1.0]
    },
    "c2_uv": {
        "1": [566.2678310305566, 389.39144015048635, 1.0],
        "2": [416.5555725226591, 336.0944213553362, 1.0],
        "3": [333.0230793005177, 424.23348877007226, 1.0],
        "4": [348.0797202440039, 237.24592248406233, 1.0],
        "5": [250.28887091510074 ,292.3544611196883,1.0],
        "6": [54.576812240039835, 237.2459391231599, 1.0],
        "7": [436.2550884736251, 140.22581937558073, 1.0],
        "8": [251.5516057997837, 232.0381253686244, 1.0],
        "9": [153.9545185283101, 172.56737594468407, 1.0],
        "10": [621.8269168105776, 116.70901227018555, 1.0],
        "11": [404.62654934772587, 119.99398599310848, 1.0],
        "12": [259.3941224529894, 177.55061671897323, 1.0]
    },
    "c1_xyz": {
        "1": [ 1.4155742470952875, 0.4718231974467605, 5.918149491107676, 1.0],
        "2": [ 0.4284754787655274, 0.10488127863923033, 6.0447479393294, 1.0],
        "3": [ -0.12664943688665475, 0.691466231626005, 6.189864386570197, 1.0],
        "4": [ -0.026431100785814454, -0.5645942640203278, 6.056445669841511, 1.0],
        "5": [ -0.6457819768262734, -0.2236318811414745, 6.862906453428839, 1.0],
        "6": [ -2.000628977496013, -0.5943181455953429, 6.375285796893046, 1.0],
        "7": [ 0.33107429351542295, -1.0301723614503198, 5.056878352820144, 1.0],
        "8": [ -0.7459566821273382, -0.4686306407995857, 4.737143191971172, 1.0],
        "9": [ -1.3743070661308023, -1.160706171818047, 7.064183439323359, 1.0],
        "10": [ 0.9233258514511951, -0.9299454459961969, 3.9817396395699136, 1.0],
        "11": [ -0.06377315102523085, -0.9338621381266773, 4.14218005113549, 1.0],
        "12": [ -0.5308583718843123, -1.2042242774086591, 7.531009849928555, 1.0]
    },
    "xyz": {
        "1": [1.4608523845672607, 6.0, -1.0191059112548828],
        "2": [0.46085238456726074, 6.0, -0.6654993891716003],
        "3": [-0.10152754187583923, 6.0, -1.2630219459533691],
        "4": [0.0, 6.0, 0.0],
        "5": [-0.7348592877388,6.662215232849121,-0.4143460988998413],
        "6": [ -2.0, 6.0, 0.0],
        "7": [ 0.5053249001502991, 5.117485523223877, 0.5563477873802185],
        "8": [ -0.498493492603302, 4.579343318939209, 0.026907440274953842],
        "9": [ -1.5, 6.829249382019043, 0.5],
        "10": [ 1.2628240585327148, 4.14644193649292, 0.5563477873802185],
        "11": [ 0.26282399892807007, 4.14644193649292, 0.5453550815582275],
        "12": [ -0.742500901222229, 7.427096843719482, 0.5]
    }
}"""

K1_gt = np.array([[8.888890245225694571e+02, 0.000000000000000000e+00, 3.200000000000000000e+02],
                  [0.000000000000000000e+00, 8.888890245225694571e+02, 2.400000000000000000e+02],
                  [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

K2_gt = np.array([[8.888890245225694571e+02, 0.000000000000000000e+00, 3.200000000000000000e+02],
                  [0.000000000000000000e+00, 8.888890245225694571e+02, 2.400000000000000000e+02],
                  [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

Rt2_gt = np.array(
    [[9.870989918708801270e-01, 1.486194040626287460e-02, -1.594200581312179565e-01, 1.000000000000000000e+00],
     [1.601113229990005493e-01, -9.162205457687377930e-02, 9.828376173973083496e-01, 0.000000000000000000e+00],
     [4.805819457942561712e-07, -9.956829547882080078e-01, -9.281960129737854004e-02, 0.000000000000000000e+00],
     [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

Rt2_gt = np.array(
    [[9.995014071464538574e-01, 9.776899969438090920e-05, -3.157362714409828186e-02, 0.000000000000000000e+00],
     [3.157377615571022034e-02, -3.096777014434337616e-03, 9.994966387748718262e-01, 0.000000000000000000e+00],
     [-5.669114955253462540e-08, -9.999951720237731934e-01, -3.098319983109831810e-03, 0.000000000000000000e+00],
     [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

#### end of ground truth data from Blender ####

# X1 = [  10.0000,  92.0000,   8.0000,  92.0000, 289.0000, 354.0000, 289.0000, 353.0000,  69.0000, 294.0000,  44.0000, 336.0000 ]
# Y1 = [ 232.0000, 230.0000, 334.0000, 333.0000, 230.0000, 278.0000, 340.0000, 332.0000,  90.0000, 149.0000, 475.0000, 433.0000 ]
# X2 = [ 123.0000, 203.0000, 123.0000, 202.0000, 397.0000, 472.0000, 398.0000, 472.0000, 182.0000, 401.0000, 148.0000, 447.0000 ]
# Y2 = [ 239.0000, 237.0000, 338.0000, 338.0000, 236.0000, 286.0000, 348.0000, 341.0000,  99.0000, 153.0000, 471.0000, 445.0000 ]

# K1 = [[844.310547,  0, 243.413315],
#       [0, 1202.508301, 281.529236],
#       [0, 0, 1]]
# K2 = [[852.721008,  0, 252.021805],
#       [0, 1215.657349, 288.587189],
#       [0, 0, 1]]

X1 = []
Y1 = []
X2 = []
Y2 = []
GT_3D_X = []
GT_3D_Y = []
GT_3D_Z = []

j = json.loads(ground_truth_points)
for key in j['c1_uv']:
    X1.append(j['c1_uv'][key][0])
    Y1.append(j['c1_uv'][key][1])
    X2.append(j['c2_uv'][key][0])
    Y2.append(j['c2_uv'][key][1])
    GT_3D_X.append(j['c1_xyz'][key][0])
    GT_3D_Y.append(j['c1_xyz'][key][1])
    GT_3D_Z.append(j['c1_xyz'][key][2])

K1 = K1_gt
K2 = K2_gt

Rz1 = np.array([[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]])
Rz2 = np.array([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])

img1 = cv2.imread('batinria0.tif', cv2.IMREAD_COLOR)
img2 = cv2.imread('batinria1.tif', cv2.IMREAD_COLOR)

########################################################################################################################
# 'raw' functions
########################################################################################################################

# Create "a" matrix for calculating E matrix from 8 point algorithm, Slide 11,12 Cremers MVG_05
def rawCreateMatrixANormalized(X1, Y1, X2, Y2):
    a = []
    for px1, py1, px2, py2 in zip(X1, Y1, X2, Y2):
        [x1, y1, z1] = np.linalg.inv(K1) @ [px1, py1, 1]
        [x2, y2, z2] = np.linalg.inv(K2) @ [px2, py2, 1]
        row = np.kron([x1, y1, z1], [x2, y2, z2])
        a.append(row)
    return a

# Create "a" matrix for calculating F matrix from 8 point algorithm, Slide 11,12 Cremers MVG_05
def rawCreateMatrixARaw(X1, Y1, X2, Y2):
    a = []
    for px1, py1, px2, py2 in zip(X1, Y1, X2, Y2):
        row = np.kron([px1, py1, 1], [px2, py2, 1])
        a.append(row)
    return a

# Recover E matrix up to sign, Slide 13,14 Cremers MVG_05
# Depending on type of input coordinates, this will rpovide E or F matrix
def rawCreateMatrixE(A):
    u, s, vt = np.linalg.svd(A)
    Es = vt[-1]
    E = Es.reshape((3, 3))
    u, s, vt = np.linalg.svd(E)
    # s = [1, 1, 0]
    s[2] = 0
    E = u @ np.diag(s) @ vt
    return E

# Recover R,T matrix from E, Slide 14 Cremers MVG_05
def rawDecomposeE(E):
    u, s, vt = np.linalg.svd(E)
    R1 = u @ np.transpose(Rz1) @ vt
    R2 = u @ np.transpose(Rz2) @ vt

    s[0] = 1
    s[1] = 1
    s[2] = 0
    T1s = u @ Rz1 @ np.diag(s) @ np.transpose(u)
    # Need to get T from skew symmetric matrix
    T = np.array([ T1s[2,1], -T1s[2,0], T1s[1,0] ]).reshape(3,1)
    return (R1, R2, T)

def mpelkaDecomposeE(E):
    u, s, vt = np.linalg.svd(E)
    R1 = u @ np.transpose(Rz1) @ vt
    R2 = u @ np.transpose(Rz2) @ vt
    T = np.array([[u[0, 2]], [u[1, 2]], [u[2, 2]]])
    return (R1, R2, T)

def _rawTriangulatePoint(P1, P2, x1, x2):
    """https://culturalengineerassociation.weebly.com/uploads/8/6/7/7/86776910/programming_computer_vision_with_python.pdf"""
    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    U,S,V = np.linalg.svd(M) # give same results for np.transpose(M) @ M
    X = V[-1,:4]

    return X / X[3]

def _rawTriangulatePoints(P1, P2, pts1, pts2):
    X = []
    for (x1,y1),(x2,y2) in zip(np.transpose(pts1),np.transpose(pts2)):
        X.append( _rawTriangulatePoint (P1,P2, np.array([x1,y1, 1]),np.array([x2,y2, 1])))
    return np.transpose(np.array(X))

# Takes P1 as projection matrix of first cam, P2 as list of 4 candidate projection matrices for second cam
def rawTriangulatePoints(P1, P2, pts1, pts2):
    # Convert pts to 2xN float arrays for OpenCV triangulate function
    pts1 = np.float32(np.transpose(pts1))
    pts2 = np.float32(np.transpose(pts2))

    # Pick the P2 matrix with the most scene points in front of the cameras after triangulation
    ind = 0
    maxres = 0
    for i in range(4):
        # Triangulate inliers and compute depth for each camera
        homog_3D = _rawTriangulatePoints(K1 @ P1, K2 @ (P2[i]), pts1, pts2)
        homog_3D = homog_3D / homog_3D[3]
        # The sign of the depth is the 3rd value of the image point after projecting back to the image
        d1 = np.dot(P1, homog_3D)[2]
        d2 = np.dot(P2[i], homog_3D)[2]
        print("i  %d :  %d %d" % (i, sum(d1 > 0), sum(d2 > 0)))
        if sum(d1 > 0) + sum(d2 > 0) > maxres:
            maxres = sum(d1 > 0) + sum(d2 > 0)
            ind = i
            infront = (d1 > 0) & (d2 > 0)

    # triangulate inliers and keep only points that are in front of both cameras
    # homog_3D is a 4xN array of reconstructed points in homogeneous coordinates, pts_3D is a Nx3 array
    homog_3D = _rawTriangulatePoints(K1 @ P1, K2 @ (P2[ind]), pts1, pts2)
    homog_3D = homog_3D[:, infront]
    homog_3D = homog_3D / homog_3D[3]
    pts_3D = np.array(homog_3D[:3]).T

    return homog_3D, pts_3D, infront

########################################################################################################################
# OPENCV functions
########################################################################################################################

# Draws epilines and points on img1 and img2
# img1 - image on which we draw the epilines for the points in img2
# lines - corresponding epilines
def drawEpilines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape[0:2]
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def drawEpilinesF(img1, img2, pts1, pts2, F):
    # Verify fundamental matrix with epiline drawing
    # Find epilines corresponding to points in right image (first image) and
    # drawing its lines on left image (second image)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawEpilines(img2, img1, lines2, pts2, pts1)

    # Find epilines corresponding to points in left image (second image) and
    # drawing its lines on right image (first image)
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawEpilines(img1, img2, lines1, pts1, pts2)

    # Draw points
    if (DEBUG_DRAW):
        plt.subplot(121), plt.imshow(img3)
        plt.subplot(122), plt.imshow(img5)
        plt.show()

def drawEpilinesF(img1, img2, pts1, pts2, F):
    # Verify fundamental matrix with epiline drawing
    # Find epilines corresponding to points in right image (first image) and
    # drawing its lines on left image (second image)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawEpilines(img2, img1, lines2, pts2, pts1)

    # Find epilines corresponding to points in left image (second image) and
    # drawing its lines on right image (first image)
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawEpilines(img1, img2, lines1, pts1, pts2)

    # Draw points
    if (DEBUG_DRAW):
        plt.subplot(121), plt.imshow(img3)
        plt.subplot(122), plt.imshow(img5)
        plt.show()

# Takes P1 as projection matrix of first cam, P2 as list of 4 candidate projection matrices for second cam
def triangulatePoints(P1, P2, pts1, pts2):
    # Convert pts to 2xN float arrays for OpenCV triangulate function
    pts1 = np.float32(np.transpose(pts1))
    pts2 = np.float32(np.transpose(pts2))

    # Pick the P2 matrix with the most scene points in front of the cameras after triangulation
    ind = 0
    maxres = 0
    for i in range(4):
        # Triangulate inliers and compute depth for each camera
        homog_3D = cv2.triangulatePoints(K1 @ P1, K2 @ (P2[i]), pts1, pts2)
        homog_3D = homog_3D / homog_3D[3]
        # The sign of the depth is the 3rd value of the image point after projecting back to the image
        d1 = np.dot(P1, homog_3D)[2]
        d2 = np.dot(P2[i], homog_3D)[2]
        print("i  %d :  %d %d" % (i, sum(d1 > 0), sum(d2 > 0)))
        if sum(d1 > 0) + sum(d2 > 0) > maxres:
            maxres = sum(d1 > 0) + sum(d2 > 0)
            ind = i
            infront = (d1 > 0) & (d2 > 0)

    # triangulate inliers and keep only points that are in front of both cameras
    # homog_3D is a 4xN array of reconstructed points in homogeneous coordinates, pts_3D is a Nx3 array
    homog_3D = cv2.triangulatePoints(K1 @ P1, K2 @ (P2[ind]), pts1, pts2)
    homog_3D = homog_3D[:, infront]
    homog_3D = homog_3D / homog_3D[3]
    pts_3D = np.array(homog_3D[:3]).T

    return homog_3D, pts_3D, infront

########################################################################################################################
# Start of Main code
########################################################################################################################

pts1 = []
pts2 = []
for px1, py1, px2, py2 in zip(X1, Y1, X2, Y2):
    pts1.append([px1, py1])
    pts2.append([px2, py2])

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# RAW version ##########################################################################################################

# Create "A" matrix for calculating F matrix from 8 point algorithm, Slide 11,12 Cremers MVG_05
A = rawCreateMatrixARaw(X2, Y2, X1, Y1)
F = rawCreateMatrixE(A)
drawEpilinesF(img1, img2, pts1, pts2, F)

Ae = rawCreateMatrixANormalized(X2, Y2, X1, Y1)
E = rawCreateMatrixE(Ae)

# Find E matrix
E1 = np.transpose(K1) @ F @ K2
E2 = np.transpose(K2) @ F @ K1

# Check that found OpenCV E fulfills the equation x1 * E * x2 = 0
print("Epipolar constraint verification using found E matrices:")
for px2, py2, px1, py1 in zip(X1, Y1, X2, Y2):
    [x1, y1, z1] = np.linalg.inv(K1) @ [px1, py1, 1]
    [x2, y2, z2] = np.linalg.inv(K2) @ [px2, py2, 1]
    err0 = np.transpose([x1, y1, 1]) @ E @ [x2, y2, 1]
    err1 = np.transpose([x1, y1, 1]) @ E1 @ [x2, y2, 1]
    err2 = np.transpose([x1, y1, 1]) @ E2 @ [x2, y2, 1]
    # print(err0, err1, err2)

#R1, R2, T = mpelkaDecomposeE(E1) # Coursera version
R1, R2, T = rawDecomposeE(E1)     # Cremers version

P0 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=np.float32)
P1 = [np.concatenate((R1, T), axis=1),
      np.concatenate((R1, -T), axis=1),
      np.concatenate((R2, T), axis=1),
      np.concatenate((R2, -T), axis=1)]

homog_3D, pts_3D, infront = rawTriangulatePoints(P0, P1, pts1, pts2)
pts_3D_T = pts_3D.T

if (DEBUG_DRAW):
    ax = plt.axes(projection="3d")
    ax.scatter3D(pts_3D_T[0], pts_3D_T[1], pts_3D_T[2])
    ax.scatter3D(GT_3D_X, GT_3D_Y, GT_3D_Z)
    plt.show()

# OpenCV version #######################################################################################################

F_opencv, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

drawEpilinesF(img1,img2,pts1,pts2, F_opencv)

# Find E matrix
E1 = np.transpose(K1) @ F_opencv @ K2
E2 = np.transpose(K2) @ F_opencv @ K1

# Check that found OpenCV E fulfills the equation x1 * E * x2 = 0
print("Epipolar constraint verification using found E matrices:")
for px2, py2, px1, py1 in zip(X1, Y1, X2, Y2):
    [x1, y1, z1] = np.linalg.inv(K1) @ [px1, py1, 1]
    [x2, y2, z2] = np.linalg.inv(K2) @ [px2, py2, 1]
    err1 = np.transpose([x1, y1, 1]) @ E1 @ [x2, y2, 1]
    err2 = np.transpose([x1, y1, 1]) @ E2 @ [x2, y2, 1]
    #print(err0, err1, err2)

R1, R2, T = cv2.decomposeEssentialMat(E1)

P0 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=np.float32)
P1 = [np.concatenate((R1, T), axis=1),
      np.concatenate((R1, -T), axis=1),
      np.concatenate((R2, T), axis=1),
      np.concatenate((R2, -T), axis=1)]

homog_3D, pts_3D, infront = triangulatePoints(P0, P1, pts1, pts2)
pts_3D_T = pts_3D.T

if (DEBUG_DRAW):
    ax = plt.axes(projection="3d")
    ax.scatter3D(pts_3D_T[0], pts_3D_T[1], pts_3D_T[2])
    ax.scatter3D(GT_3D_X, GT_3D_Y, GT_3D_Z)
    plt.show()
