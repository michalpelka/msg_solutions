import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, logm
from scipy import interpolate
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
    Kd[2][2]=1
    return [Id,Dd,Kd]

def calcErrPoint(I1, D1, I2, se3xi, K , u, v):
    R = se3xi[0:3,0:3]
    T = se3xi[0:3,3]
    cx = K[0][2]; cy= K[1][2]
    fx = K[0][0]; fy= K[1][1]
    if D1[int(u)][int(v)] == 0:
        return 0
    p3d_frame1 = np.array([(u - cx) / fx,(v - cy) / fy, 1]) *  D1[int(u)][int(v)]
    p3d_frame2 = R @ p3d_frame1 + T
    p3d_frame2 = p3d_frame2 / p3d_frame2[2]
    up = (fx * p3d_frame2[0] + cx)
    vp = (fy * p3d_frame2[1] + cy)

    if (up > 0 and up < I1.shape[0]):
        if (vp > 0 and vp < I1.shape[1]):
            res = (I1[int(u)][int(v)]-I2(up,vp))
            #res = I2(up,vp)
            return res*res
    # n1 = cv2.undistortPoints(np.array([(x,y)]), np.array(K), np.array([0.0]*4))
    # n11 = [(u - cx) / fx,(v - cy) / fy, 1]
    
    # n1 = np.array([n1[0][0][0], n1[0][0][1],1]) * D1[int(x)][int(y)]
    # X = np.array([[1,0,0,xi[3]],[0,1,0,xi[4]],[0,0,0,xi[5]],[0,0,0,1]])
    # X[0:3,0:3] = skew (xi[0:3])
    # n2 = X @ [n1[0],n1[1],n1[2],1]
    return 0

def calcPoint(I1, D1, I2, se3xi, K , u, v):
    cx = K[0][2]; cy= K[1][2]
    fx = K[0][0]; fy= K[1][1]
    R = se3xi[0:3,0:3]
    T = se3xi[0:3,3]

    if D1[int(u)][int(v)] == 0:
        return 0
    p3d_frame1 = np.array([(u - cx) / fx,(v - cy) / fy, 1]) *  D1[int(u)][int(v)]
    p3d_frame2 = R @ p3d_frame1 + T
    p3d_frame2 = p3d_frame2 / p3d_frame2[2]
    up = (fx * p3d_frame2[0] + cx)
    vp = (fy * p3d_frame2[1] + cy)

    if (up > 0 and up < I1.shape[0]):
        if (vp > 0 and vp < I1.shape[1]):
            res = I2(up,vp)
            #res = I2(up,vp)
            return res
    return 0  
def calcP(I1, D1, I2, xi, K):
    assert (I1.shape==I2.shape)
    assert (I1.shape==D1.shape)
    assert (len(xi)==6)
    f_I2 = interpolate.interp2d(np.arange(0,I2.shape[0]), np.arange(0,I2.shape[1]), np.transpose(I2) )
    res_img = np.zeros(I1.shape)
    se3xi = se3exp(xi)
    for u in range (0,I1.shape[0]):
        for v in range (0,I2.shape[1]):
            res_img[u][v]=calcPoint(I1, D1, f_I2, se3xi, K ,u, v)
    return res_img

def calcErr(I1, D1, I2, xi, K):
    assert (I1.shape==I2.shape)
    assert (I1.shape==D1.shape)
    assert (len(xi)==6)
    f_I2 = interpolate.interp2d(np.arange(0,I2.shape[0]), np.arange(0,I2.shape[1]), np.transpose(I2) )
    res_img = np.zeros(I1.shape)
    se3xi = se3exp(xi)
    for u in range (0,I1.shape[0]):
        for v in range (0,I2.shape[1]):
            res_img[u][v]=calcErrPoint(I1, D1, f_I2, se3xi, K , u,v)
    return res_img

def deriveNumeric(I1, D1, I2, xi, K):
    assert (len(xi)==6)
    J = []
    for i in range (0,6):
        e = [0] * 6
        EPS = 10e-6
        e[i] = EPS
        xi2 = se3log (se3exp(e) @ se3exp(xi))
        #print (xi2, xi)
        err2 = calcErr(I1,D1,I2, xi2, K)
        err1 = calcErr(I1,D1,I2, xi, K)
        d = (err2-err1) / EPS
        #plt.subplot(2,3,i+1)
        #plt.imshow(d)
        #plt.colorbar()
        J.append(d.reshape(-1))
    #plt.show()
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

PYRAMID_LEVELS = 4

pyr1 = [[c1, d1, K]]
pyr2 = [[c2, d2, K]]

for i in range (PYRAMID_LEVELS):
    pyr1.append(downscale ( *pyr1[-1] ))
    pyr2.append(downscale ( *pyr2[-1] ))


#sp = [ -0.0292,   -0.0183,   -0.0009, -0.0021,    0.0057,    0.0374]
sp = [0]*6
[Id1,Dd1,Kd1] = pyr1[2]
[Id2,Dd2,Kd2] = pyr2[2]
#plt.imshow(calcP(Id1, Dd1, Id2, sp, Kd1));plt.savefig("/tmp/gt_init.png")

with open("/tmp/optim.txt", 'w') as log:
    for k in range (PYRAMID_LEVELS, 0, -1):
        [Id1,Dd1,Kd1] = pyr1[k]
        [Id2,Dd2,Kd2] = pyr2[k]
        for i in range (0,6):
            r = calcErr(Id1, Dd1, Id2, sp, Kd1)
            J = deriveNumeric(Id1, Dd1, Id2,sp, Kd1)
            err = np.sum(r)
            delta_sp = np.linalg.inv(-(np.transpose(J) @ J)) @ np.transpose(J) @ r.reshape(-1,1)
            sp = se3log (se3exp(delta_sp) @ se3exp(sp))
            print ("%0.1f  %s"%(err, np.around(sp,decimals=4)))
            log.write("%0.1f  %s\n"%(err, np.around(sp,decimals=4)))
        plt.imshow(calcP(c1, d1, c2, sp, K));plt.savefig("/tmp/pc1_pyr%d.png"%(k))

#plt.subplot(121)

plt.imshow(c2);plt.savefig("/tmp/c2.png")
plt.imshow(c1);plt.savefig("/tmp/c1.png")
#res_img = calcErr(Id1, Dd1, Id2, [0.1,0,0,0,0,0], K)
# plt.subplot(121)
# plt.imshow(Id1)
# plt.subplot(122)
# plt.imshow(res_img)
#plt.colorbar()
# plt.show()