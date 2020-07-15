import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def skew(a):
    return np.array (((0,    - a[2],   a[1]),
                     (a[2],   0  ,  -a[0]),
                     (-a[1],  a[0],   0)))
                     
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

def calcErr(I1, D1, I2, xi, K , u, v):
    cx = K[0][2]; cy= K[1][2]
    fx = K[0][0]; fy= K[1][1]
    n1 = np.array([(u - cx) / fx,(v - cy) / fy, 1]) *  D1[int(u)][int(v)]
    n2 = (np.identity(3) + skew (xi)) @ n1 + xi[3:]
    n2n = n2 / n2[2]
    up = fx * n2n[0] + cx
    vp = fy * n2n[1] + cy
    if (up > 0 and up < I1.shape[0]):
        if (vp > 0 and vp < I1.shape[1]):
            error = I1[int(u)][int(v)]-I2[int(up)][int(vp)]
            return  error*error
    # n1 = cv2.undistortPoints(np.array([(x,y)]), np.array(K), np.array([0.0]*4))
    # n11 = [(u - cx) / fx,(v - cy) / fy, 1]
    
    # n1 = np.array([n1[0][0][0], n1[0][0][1],1]) * D1[int(x)][int(y)]
    # X = np.array([[1,0,0,xi[3]],[0,1,0,xi[4]],[0,0,0,xi[5]],[0,0,0,1]])
    # X[0:3,0:3] = skew (xi[0:3])
    # n2 = X @ [n1[0],n1[1],n1[2],1]
    return 0
    
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

[Id,Dd,Kd] = downscale ( c1, d1, K)
[Id,Dd,Kd] = downscale ( Id,Dd,Kd)


rots = np.arange(0.0, 0.5, 0.01)
errors = []
for rot in rots:
    e = 0
    for u in range (0,Id.shape[0]):
        for v in range (0,Id.shape[1]):
            e+=calcErr(Id, Dd, Id, [0,rot,0,0,0,0], Kd , u,v)
    errors.append(e)
plt.plot(rots,errors)
plt.show()