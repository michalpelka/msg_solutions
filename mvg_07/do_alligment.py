import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, logm
def skew(a):
    return np.array (((0,    -a[2],   a[1]),
                     (a[2],   0  ,  -a[0]),
                     (-a[1],  a[0],   0)))
def se3exp(a):
    T=  np.array (((   0,    -a[2],   a[1],   a[3]),
                   ( a[2],      0 ,  -a[0],   a[4]),
                   (-a[1],    a[0],      0,   a[5]),
                   (     0,      0,      0,   0)))
    return expm(T)
def se3log(a):
    lg = logm(a)
    t = [lg[2,1], lg[0,2], lg[1,0], lg[0,3],lg[1,3], lg[2,3]]
    return t

                     
def downscale( I, D, K):
    h = int(I.shape[0]/2)
    w = int(I.shape[1]/2)
    Id = np.zeros((h,w))
    for y in range(0, w):
        for x in range(0, h):
            Id[x][y] = 0.25 * I[2*x+1][2*y+1]+ \
                0.25 * I[2*x+1][2*y]+ \
                0.25 * I[2*x][2*y+1]+ \
                0.25 * I[2*x][2*y]
    Dd = np.zeros((h,w))
    for y in range(0, w):
        for x in range(0, h):
            Dd[x][y] = 0
            p = 0.0
            if D[2*x+1][2*y+1] != 0:
                Dd[x][y] += 1.0/D[2*x+1][2*y+1]
                p+=1.0
            if D[2*x+1][2*y] != 0:
                Dd[x][y] += 1.0/D[2*x+1][2*y]
                p+=1.0
            if D[2*x][2*y+1] != 0:
                Dd[x][y] += 1.0/D[2*x][2*y+1]
                p+=1
            if D[2*x][2*y] != 0:
                Dd[x][y] += 1.0/D[2*x][2*y]
                p+=1.0
            if p > 0:
                Dd[x][y]  = Dd[x][y]/p
                Dd[x][y] = 1.0/Dd[x][y] 
    Kd = np.array(K)/2
    return [Id,Dd,Kd]

def calcErrPoint(I1, D1, I2, xi, K , u, v):
    cx = K[0][2]; cy= K[1][2]
    fx = K[0][0]; fy= K[1][1]
    if D1[int(u)][int(v)] == 0:
        return 0
    n1 = np.array([(u - cx) / fx,(v - cy) / fy, 1]) *  D1[int(u)][int(v)]
    n2 = (np.identity(3) + skew (xi)) @ n1 + xi[3:]
    n2n = n2 / n2[2]
    up = int(round(fx * n2n[0] + cx))
    vp = int(round(fy * n2n[1] + cy))
    if (up > 0 and up < I1.shape[0]):
        if (vp > 0 and vp < I1.shape[1]):
            res = (I1[int(u)][int(v)]-I2[up][vp])
            return res*res
    # n1 = cv2.undistortPoints(np.array([(x,y)]), np.array(K), np.array([0.0]*4))
    # n11 = [(u - cx) / fx,(v - cy) / fy, 1]
    
    # n1 = np.array([n1[0][0][0], n1[0][0][1],1]) * D1[int(x)][int(y)]
    # X = np.array([[1,0,0,xi[3]],[0,1,0,xi[4]],[0,0,0,xi[5]],[0,0,0,1]])
    # X[0:3,0:3] = skew (xi[0:3])
    # n2 = X @ [n1[0],n1[1],n1[2],1]
    return 0
def calcErr(I1, D1, I2, xi, K):
    assert (I1.shape==I2.shape)
    assert (I1.shape==D1.shape)
    assert (len(xi)==6)
    res_img = np.zeros(I1.shape)
    for u in range (0,I1.shape[0]):
        for v in range (0,I2.shape[1]):
            res_img[u][v]=calcErrPoint(I1, D1, I2, xi, K , u,v)
    return res_img

def deriveNumeric(I1, D1, I2, xi, K):
    assert (len(xi)==6)
    J = []
    for i in range (0,6):
        e = [0] * 6
        EPS = 10e-4
        e[i] = EPS
        xi2 = se3log (se3exp(e) @ se3exp(xi))
        err2 = calcErr(I1,D1,I2, xi2, K)
        err1 = calcErr(I1,D1,I2, xi, K)
        d = (err2-err1) / EPS
        plt.subplot(2,3,i+1)
        plt.imshow(d)
        J.append(d.reshape(-1))
    plt.show()
    return np.transpose(np.stack(J))

K = [[517.3, 0, 318.6],	[0, 516.5, 255.3,], [0, 0, 1]]
_c2 = cv2.imread('rgb/1305031102.175304.png')
_c1 = cv2.imread('rgb/1305031102.275326.png')
c1 = cv2.cvtColor(_c1,cv2.COLOR_BGR2GRAY)
c2 = cv2.cvtColor(_c2,cv2.COLOR_BGR2GRAY)

d2 = cv2.imread('depth/1305031102.160407.png', cv2.IMREAD_UNCHANGED)/5000
#d2 = d2[:,:,1]
d1 = cv2.imread('depth/1305031102.262886.png', cv2.IMREAD_UNCHANGED)/5000
#d1 = d1[:,:,1]
# result:
# approximately -0.0021    0.0057    0.0374   -0.0292   -0.0183   -0.0009

##
# K = [ [535.4,  0, 320.1],[0, 539.2, 247.6],[0, 0, 1]]
# c1 = cv2.imread('rgb/1341847980.722988.png')
# c2 = cv2.imread('rgb/1341847982.998783.png')
# #c1 = cv2.cvtColor(c1,cv2.COLOR_BGR2GRAY)
# #c2 = cv2.cvtColor(c2,cv2.COLOR_BGR2GRAY)

# d1 = cv2.imread('depth/1341847980.723020.png')/5000
# d2 = cv2.imread('depth/1341847982.998830.png')/5000

# result:
#  approximately -0.2894 0.0097 -0.0439  0.0039 0.0959 0.0423

[Id1,Dd1,Kd1] = downscale ( c1, d1, K)
[Id1,Dd1,Kd1] = downscale ( Id1, Dd1, Kd1)

[Id2,Dd2,Kd2] = downscale ( c2, d2, K)
[Id2,Dd2,Kd2] = downscale ( Id2, Dd2, Kd2)

#res_img = calcErr(Id1, Dd1, Id2, [0,0,0,0,0,0], K)
J = deriveNumeric(Id1, Dd1, Id2, [0,0,0,0,0,0], K)

plt.imshow(res_img)
plt.colorbar()
plt.show()