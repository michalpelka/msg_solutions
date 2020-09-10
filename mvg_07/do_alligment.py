import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, logm
from scipy import interpolate

def skew(a):
    return np.array (((   0, -a[5],    a[4]),
                      (a[5],    0  ,  -a[3]),
                     (-a[4],  a[3],    0)))

def se3exp(a):
    T = np.array (((   0,    -a[5],   a[4],   a[0]),
                   ( a[5],      0 ,  -a[3],   a[1]),
                   (-a[4],    a[3],      0,   a[2]),
                   (    0,       0,      0,      0)))
    return expm(T)
    
def se3log(a):
    lg = logm(a)
    #lg(1,4) lg(2,4) lg(3,4) lg(3,2) lg(1,3) lg(2,1)
    t = [lg[0,3], lg[1,3], lg[2,3], lg[2,1],lg[0,2], lg[1,0]]
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
                Dd[x][y] = Dd[x][y]/p
                Dd[x][y] = 1.0/Dd[x][y] 
    Kd = np.array(K)
    Kd[0][0] = Kd[0][0]/2
    Kd[1][1] = Kd[1][1]/2
    Kd[0][2] = (Kd[0][2] + 0.5) / 2.0 - 0.5
    Kd[1][2] = (Kd[1][2] + 0.5) / 2.0 - 0.5
    Kd[2][2]=1
    return [Id,Dd,Kd]

def calcErrPoint(I1, D1, I2, se3xi, K , u, v):
    R = se3xi[0:3,0:3]
    T = se3xi[0:3,3]
    cx = K[0][2]; cy = K[1][2]
    fx = K[0][0]; fy = K[1][1]
    if D1[int(u)][int(v)] == 0:
        return 0
    p3d_frame1 = np.array([(u - cx) / fx,(v - cy) / fy, 1]) * D1[int(u)][int(v)]
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
    cx = K[0][2]; cy = K[1][2]
    fx = K[0][0]; fy = K[1][1]
    R = se3xi[0:3,0:3]
    T = se3xi[0:3,3]

    if D1[int(u)][int(v)] == 0:
        return 0
    p3d_frame1 = np.array([(u - cx) / fx,(v - cy) / fy, 1]) * D1[int(u)][int(v)]
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

def getTransformedImg(I1, D1, I2, xi, K):
    assert (I1.shape==I2.shape)
    assert (I1.shape==D1.shape)
    assert (len(xi)==6)
    f_I2 = interpolate.interp2d(np.arange(0,I2.shape[0]), np.arange(0,I2.shape[1]), np.transpose(I2))
    img2_warped_to_im1 = np.zeros(I1.shape)
    se3xi = se3exp(xi)
    for u in range (0,I1.shape[0]):
        for v in range (0,I1.shape[1]):
            img2_warped_to_im1[u][v]=calcPoint(I1, D1, f_I2, se3xi, K, u, v)
    return img2_warped_to_im1

def getResidualImg(I1, D1, I2, xi, K):
    assert (I1.shape==I2.shape)
    assert (I1.shape==D1.shape)
    assert (len(xi)==6)
    f_I2 = interpolate.interp2d(np.arange(0,I2.shape[0]), np.arange(0,I2.shape[1]), np.transpose(I2))
    res_img = np.zeros(I1.shape)
    se3xi = se3exp(xi)
    for u in range (0,I1.shape[0]):
        for v in range (0,I1.shape[1]):
            res_img[u][v]=calcErrPoint(I1, D1, f_I2, se3xi, K, u, v)
    return res_img

def deriveAnalytic(I1, D1, I2, xi, K, dI2x, dI2y):
    assert (len(xi)==6)
    J = []
    cx = K[0][2]; cy = K[1][2]
    fx = K[0][0]; fy = K[1][1]
    R = se3exp(xi)[0:3,0:3]
    T = se3exp(xi)[0:3,3]
    f_dI2x = interpolate.interp2d(np.arange(0,dI2x.shape[0]), np.arange(0,dI2x.shape[1]), np.transpose(dI2x))
    f_dI2y = interpolate.interp2d(np.arange(0,dI2y.shape[0]), np.arange(0,dI2y.shape[1]), np.transpose(dI2y))
    cv2.imwrite("/tmp/dI2x_pyt.png", dI2x)
    cv2.imwrite("/tmp/dI2y_pyt.png", dI2y)
    
    for u in range (0,I1.shape[0]):
        for v in range (0,I1.shape[1]):
            p3d_frame1 = np.array([(u - cx) / fx, (v - cy) / fy, 1]) * D1[int(u)][int(v)]
            p3d_frame2 = R @ p3d_frame1 + T
            [xp,yp,zp] = [np.nan, np.nan, np.nan]
            up = np.nan
            vp = np.nan            
            if p3d_frame2[2] != 0 and  D1[int(u)][int(v)] != 0:
                [xp,yp,zp] = p3d_frame2
                p3d_frame2 = p3d_frame2 / p3d_frame2[2]
                #pTransProj = K @ p3d_frame2
                #up = pTransProj[0]/pTransProj[2]
                #vp = pTransProj[]/pTransProj[2]
                
                up = (fx * p3d_frame2[0] + cx)
                vp = (fy * p3d_frame2[1] + cy)

            Jw = np.array([np.nan]*6)

            dxInterp = fx*float(f_dI2x(up, vp))
            dyInterp = fy*float(f_dI2y(up, vp))
            Jw[0] = dxInterp / zp
            Jw[1] = dyInterp / zp
            Jw[2] = - (dxInterp * xp + dyInterp * yp) / (zp * zp)
            Jw[3] = - (dxInterp * xp * yp) / (zp * zp) - dyInterp * (1 + (yp / zp)*(yp / zp))
            Jw[4] = + dxInterp * (1 + (xp / zp)*(xp / zp)) + (dyInterp * xp * yp) / (zp * zp)
            Jw[5] = (- dxInterp * yp + dyInterp * xp) / zp
            Jw = -Jw
            # Jw = -1.0/zp * np.array([float(f_dI1x(u2,v2)*fx), float(f_dI1y(u2,v2)*fy)])
            # Jw = Jw @ np.array([[1, 0, -xp/zp, -xp*yp/zp, (zp + xp*xp/zp), -yp  ],
            #                     [0, 1, -yp/zp, -(zp + yp*yp/zp), xp*yp/zp,  xp  ]])
            J.append(Jw)

    J = np.stack(J)
    
    # for i in range(6):
    #     plt.subplot(2,3,i+1)
    #     plt.imshow(J[:,i].reshape(I1.shape))
    #     plt.colorbar()
    # plt.show()
    return J 

    
def deriveNumeric(I1, D1, I2, xi, K):
    assert (len(xi)==6)
    J = []
    for i in range (0,6):
        e = [0] * 6
        EPS = 10e-8
        e[i] = EPS
        xi2 = se3log(se3exp(e) @ se3exp(xi))
        #print (xi2, xi)
        err2 = getResidualImg(I1, D1, I2, xi2, K)
        err1 = getResidualImg(I1, D1, I2, xi, K)
        d = (err2-err1) / EPS
        #plt.subplot(2,3,i+1)
        #plt.imshow(d)
        #plt.colorbar()
        J.append(d.reshape(-1))
    #plt.show()
    J =  np.transpose(np.stack(J))
    # for i in range(6):
    #     plt.subplot(2,3,i+1)
    #     plt.imshow(J[:,i].reshape(I1.shape))
    #     plt.colorbar()
    # plt.show()

    return J

#dataset 1
K = [[517.3, 0, 318.6],	[0, 516.5, 255.3,], [0, 0, 1]]
c2 = cv2.imread('rgb/1305031102.175304.png')
c1 = cv2.imread('rgb/1305031102.275326.png')
c1 = cv2.cvtColor(c1,cv2.COLOR_BGR2GRAY)
c2 = cv2.cvtColor(c2,cv2.COLOR_BGR2GRAY)

d2 = cv2.imread('depth/1305031102.160407.png', cv2.IMREAD_UNCHANGED)/5000
d1 = cv2.imread('depth/1305031102.262886.png', cv2.IMREAD_UNCHANGED)/5000

#dataset 2
# K = [ [535.4,  0, 320.1],[0, 539.2, 247.6],[0, 0, 1]]
# c1 = cv2.imread('rgb/1341847980.722988.png')
# c2 = cv2.imread('rgb/1341847982.998783.png')
# c1 = cv2.cvtColor(c1,cv2.COLOR_BGR2GRAY)
# c2 = cv2.cvtColor(c2,cv2.COLOR_BGR2GRAY)

# d1 = cv2.imread('depth/1341847980.723020.png', cv2.IMREAD_UNCHANGED)/5000;
# d2 = cv2.imread('depth/1341847982.998830.png', cv2.IMREAD_UNCHANGED)/5000;

#result:
#  approximately -0.2894 0.0097 -0.0439  0.0039 0.0959 0.0423

PYRAMID_LEVELS = 4

pyr1 = [[c1, d1, K]]
pyr2 = [[c2, d2, K]]

for i in range (PYRAMID_LEVELS):
    pyr1.append(downscale( *pyr1[-1] ))
    pyr2.append(downscale( *pyr2[-1] ))


#sp = [ -0.0292,   -0.0183,   -0.0009, -0.0021,    0.0057,    0.0374]
twist_estimate = [0] * 6
# [Id1,Dd1,Kd1] = pyr1[2]
# [Id2,Dd2,Kd2] = pyr2[2]
#plt.imshow(calcP(Id1, Dd1, Id2, sp, Kd1));plt.savefig("/tmp/gt_init.png")
plt.imshow(c2);#plt.savefig("/tmp/c2.png")
plt.imshow(c1);#plt.savefig("/tmp/c1.png")

with open("/tmp/optim.txt", 'w') as log:
    for k in range (PYRAMID_LEVELS, 0, -1):

        [Id1,Dd1,Kd1] = pyr1[k]
        [Id2,Dd2,Kd2] = pyr2[k]
        dI2x = cv2.Sobel(Id2, cv2.CV_64F, 1, 0, ksize=1)
        dI2y = cv2.Sobel(Id2, cv2.CV_64F, 0, 1, ksize=1)
        
        plt.imshow(dI2x);#plt.savefig("/tmp/dI2x.png")
        plt.imshow(dI2y);#plt.savefig("/tmp/dI2y.png")

        for i in range (0,5):
            #J = deriveNumeric(Id1, Dd1, Id2, twist_estimate, Kd1)
            J = deriveAnalytic(Id1, Dd1, Id2, twist_estimate, Kd1, dI2x, dI2y)
            r = getResidualImg(Id1, Dd1, Id2, twist_estimate, Kd1)
            nans=~np.isnan(J).any(axis=1)
            J_trim = J[nans]
            r_trim = r.reshape(-1,1)[nans]
            
            err = np.sum(r)
            twist_estimate_delta = np.linalg.inv(-(np.transpose(J_trim) @ J_trim)) @ np.transpose(J_trim) @ r_trim
            twist_estimate = se3log (se3exp(twist_estimate_delta) @ se3exp(twist_estimate))

            print ("%0.1f  %s" % (err, np.around(twist_estimate, decimals=4)))
            log.write("%0.1f  %s\n" % (err, np.around(twist_estimate, decimals=4)))
            plt.imshow(r)
            plt.savefig("/tmp/err_%02d_%02d.png"%(k,i))

        plt.imshow(getTransformedImg(c1, d1, c2, twist_estimate, K));plt.savefig("/tmp/pc1_pyr%d.png" % (k))
        plt.show()
