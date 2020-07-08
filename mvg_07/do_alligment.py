import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    Kd = K
    return [Id,Dd,Kd]
K = [[517.3, 0, 318.6],	[0, 516.5, 255.3,], [0, 0, 1]]
_c2 = cv2.imread('rgb/1305031102.175304.png')
_c1 = cv2.imread('rgb/1305031102.275326.png')
c1 = cv2.cvtColor(_c1,cv2.COLOR_BGR2GRAY)
c2 = cv2.cvtColor(_c2,cv2.COLOR_BGR2GRAY)

d2 = cv2.imread('depth/1305031102.160407.png')/5000
d2 = d2[:,:,1]
d1 = cv2.imread('depth/1305031102.262886.png')/5000
d1 = d1[:,:,1]
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

[Id,Dd,Kd] = downscale ( c1, d2, K)

plt.imshow(Id)
plt.show()