import bpy
import sys
import argparse
 
parser = argparse.ArgumentParser()
if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]

parser.add_argument('--camera', type= str)
parser.add_argument('--path', type= str)

args = parser.parse_args(argv)
bpy.context.scene.cycles.samples = 320
bpy.context.scene.camera = bpy.context.scene.objects[args.camera]
bpy.context.scene.render.filepath = args.path
print ('DONE!!!!')
