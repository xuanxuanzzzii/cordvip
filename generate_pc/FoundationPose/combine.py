import trimesh

scene = trimesh.load('./object_mesh/box/box_1.obj')
meshes = list(scene.geometry.values())  
num_meshes = len(meshes)  

mesh1 = meshes[0]
mesh2 = meshes[1]
mesh3 = meshes[2]

combined_mesh = trimesh.util.concatenate([mesh1, mesh2, mesh3])

combined_mesh.show()
combined_mesh.export('./object_mesh/box/box.obj')
