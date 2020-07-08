import bpy
import math
import numpy as np
import mathutils
import json

def getCameraMatrixIntrinsic(obj):
    # get resolution:
    res_x = bpy.context.scene.render.resolution_x * bpy.context.scene.render.resolution_percentage / 100.0
    res_y = bpy.context.scene.render.resolution_y * bpy.context.scene.render.resolution_percentage / 100.0
    # get sensor size
    sensor_width_in_mm = obj.data.sensor_width
    sensor_height_in_mm = obj.data.sensor_height
    f_mm =  obj.data.lens
    f_x_px = res_x * f_mm / sensor_width_in_mm
    #f_y_px = res_y * f_mm / sensor_height_in_mm
    print (sensor_width_in_mm, sensor_height_in_mm)
    camera_matrix = np.array(((f_x_px, 0       , res_x/2),
                             (0,      f_x_px  , res_y/2),
                             (0,0,1)))
    return camera_matrix


def skew(a):
    return np.array (((0,    - a[2],   a[1]),
                     (a[2],   0  ,  -a[0]),
                     (-a[1],  a[0],   0)))
                     
np.set_printoptions(suppress=True)
blender_camera = mathutils.Matrix([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])


c1 = bpy.context.scene.objects['camera000']
c2 = bpy.context.scene.objects['camera001']

K1 = getCameraMatrixIntrinsic(c1)
K2 = getCameraMatrixIntrinsic(c2)
 
for c in ['c1_co', 'c2_co', 'c1_in_c2', 'c2_in_c1' ]:
    if c not in  bpy.context.scene.objects:
        o = bpy.data.objects.new( c, None )
        bpy.context.scene.collection.objects.link( o )
        o.empty_display_type='ARROWS'
        
c1_co = bpy.context.scene.objects['c1_co']
c2_co = bpy.context.scene.objects['c2_co']

c1_co.matrix_world = c1.matrix_world @ blender_camera
c2_co.matrix_world = c2.matrix_world @ blender_camera

c1_in_c2 = bpy.context.scene.objects['c1_in_c2']
c2_in_c1 = bpy.context.scene.objects['c2_in_c1']

c1_in_c2.parent = c2_co
c2_in_c1.parent = c1_co

c1_in_c2.matrix_local =  (c2.matrix_world @ blender_camera).inverted()  @ c1.matrix_world @ blender_camera
c2_in_c1.matrix_local = (c1.matrix_world @ blender_camera).inverted()  @ c2.matrix_world @ blender_camera

#m_local = (c2.matrix_world @ blender_camera).inverted()  @ (c1.matrix_world @ blender_camera)

c1_loc_mat = np.array(c1_in_c2.matrix_local)


R = c1_loc_mat[0:3,0:3]
T = c1_loc_mat[:3,3]
#print (T)
Tx = skew(T)
#print (Tx)

E = Tx @ R
F = np.linalg.inv(np.transpose(K1)) @ Tx @ R  @ np.linalg.inv(K2)

print ("Fundamental:")
np.savetxt("data/fundamental_groundtruth.txt",F)
np.savetxt("data/essential_groundtruth.txt",E)


np.savetxt("data/camera000_K.txt",K1)
np.savetxt("data/camera001_K.txt",K2)

np.savetxt("data/camera000_Rt.txt",c1_co.matrix_world)
np.savetxt("data/camera001_Rt.txt",c2_co.matrix_world)


print (F)

obj = {}

c1_inv_m = np.linalg.inv(np.array(c1_co.matrix_local))
c2_inv_m = np.linalg.inv(np.array(c2_co.matrix_local))
obj['xyz'] = {}
obj['c1_xyz'] = {}
obj['c2_xyz'] = {}

obj['c1_uv'] = {}
obj['c2_uv'] = {}

for object in bpy.context.scene.objects:
    if object.pass_index > 0 :
        p_xyz1= np.array([ object.location.x, object.location.y, object.location.z , 1 ])
        obj['xyz'][object.pass_index] = [ object.location.x, object.location.y, object.location.z ]
        obj['c1_xyz'][object.pass_index]  = (c1_inv_m @ p_xyz1).tolist()
        obj['c2_xyz'][object.pass_index]  = (c2_inv_m @ p_xyz1).tolist()
    
        uv1 = (K1 @ (c1_inv_m @ p_xyz1)[0:3])
        uv1 /=uv1[2]
        
        uv2 = (K2 @ (c2_inv_m @ p_xyz1)[0:3])
        uv2 /=uv2[2]
        
        obj['c1_uv'][object.pass_index]  = (uv1).tolist()
        obj['c2_uv'][object.pass_index]  = (uv2).tolist()
        
        #f.write("%d : %f %f %f\n" %(object.pass_index, object.location.x, object.location.y, object.location.z))
 
with open ("data/points.json", 'w') as f:
    json.dump(obj, f, indent=4, sort_keys=True, )   
