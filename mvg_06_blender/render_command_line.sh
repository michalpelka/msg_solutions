#! /bin/sh
mkdir -p data
blend_file=scene.blend
blender $blend_file -b --python export_fundamental.py
blender $blend_file -b --python set_camera.py -f 0 -- --camera camera000 --path data/camera000
blender $blend_file -b --python set_camera.py -f 0 -- --camera camera001 --path data/camera001
