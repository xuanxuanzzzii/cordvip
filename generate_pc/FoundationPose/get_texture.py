import bpy
import os
import math

import argparse
import sys
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

argv = sys.argv
argv = argv[argv.index("--") + 1:]

parser = argparse.ArgumentParser(description="Blender script example using argparse.")
parser.add_argument('--obj_path', type=str, required=True) 
parser.add_argument('--texture_image_path', type=str, required=True)    
parser.add_argument('--output_dir', type=str, required=True)  
args = parser.parse_args(argv)

print(f"OBJ Path: {args.obj_path}")
print(f"Texture Path: {args.texture_image_path}")
print(f"Output Directory: {args.output_dir}")

obj_path = args.obj_path
texture_image_path = args.texture_image_path
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bpy.ops.import_scene.obj(filepath=obj_path)
material = bpy.data.materials.new(name="MyMaterial")
material.use_nodes = True 

texture_node = material.node_tree.nodes.new('ShaderNodeTexImage')
texture_node.image = bpy.data.images.load(texture_image_path)
principled_node = material.node_tree.nodes.get('Principled BSDF')
material.node_tree.links.new(texture_node.outputs['Color'], principled_node.inputs['Base Color'])
obj = bpy.context.selected_objects[0]

if obj.type == 'MESH':  
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

scale_factor = 0.14
obj.scale = (scale_factor, scale_factor, scale_factor)  
bpy.context.scene.render.engine = 'CYCLES'  

output_obj_path = os.path.join(output_dir, 'mesh_with_texture.obj')
output_mtl_path = os.path.join(output_dir, 'mesh_with_texture.mtl')

bpy.ops.export_scene.obj(
    filepath=output_obj_path,
    use_materials=True,  
    path_mode='COPY', 
    use_triangles=True  
)

print(f"OBJ file exported to {output_obj_path}")
print(f"MTL file exported to {output_mtl_path}")
