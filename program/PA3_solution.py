# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 08:31:21 2021

@author: Ding
"""

import numpy as np
import PA1_functions as PAF1
import PA3_functions as PAF3
from scipy import spatial
"""
step_1_rigid_registration function returns the position of pointer tip with respect to
rigid body B
    Input:
        samples:  the SampleReadings file, in this file including the A
        body LED markers in tracker coordinates, B body LED markers in 
        tracker coordinates in different frames.
        
        A_marker:coordinates of marker LEDs in A body coordinates
        
        B_marker:coordinates of marker LEDs in B body coordinates
        
        N_frames: number of frames in samples, a default value is 15
        tip_save_path: the path to save d_{k}
        
    Output:
        tip_positions: the position of d_{k}
"""
def step_1_rigid_registration(samples, A_marker,B_marker, N_frames= 15,tip_save_path=None):
    tip_positions=[]
    A_RB=A_marker[0:6,:]
    B_RB=B_marker[0:6,:]
    A_tip=A_marker[6,:]
    for n_frame in range(N_frames):
        current_A_transformed= samples[(n_frame*16):(n_frame*16+6),:]
        current_B_transformed= samples[(n_frame*16+6):(n_frame*16+12),:]
        
        current_F_A=PAF1.points_registeration(A_RB,current_A_transformed)
        current_F_B=PAF1.points_registeration(B_RB,current_B_transformed)
                
        current_F_B_inverse=PAF1.frame_inverse(current_F_B[:,0:3],current_F_B[:,3])
        
        current_F_B_A= PAF1.frame_composition(current_F_B_inverse[:,0:3],current_F_B_inverse[:,3],current_F_A[:,0:3],current_F_A[:,3])
    
        current_position_A_tip=PAF1.frame_transformation(current_F_B_A[:,0:3],current_F_B_A[:,3],A_tip.reshape(3,1))
                
        tip_positions.append(current_position_A_tip)
        
    tip_positions=np.array(tip_positions)
    tip_positions=np.squeeze(tip_positions)
    np.save(tip_save_path,tip_positions)
    
    return tip_positions
'''
simple bounding method for searching nearest points in the surface:
    Input:
        tip_positons: the original positions of points to be searched with
        
        vertices: An Nx3 array, where each row is a vertice XYZ-coordinate
    
        triangle: An Nx3 array, where each row corresponding to the index of three
        vertices in variable 'vertices'
        
    Output:
       closest_point_set: the found point positions in surface 
'''
def step_2_simple_search(point_set, vertices, triangles):
    bound_L, bound_U=PAF3.bound_fix(vertices, triangles)
    closest_point_set=[]
    for i in range(point_set.shape[0]):
        current_point=point_set[i,:]
        current_closest_point,_,_=PAF3.simple_search(current_point, vertices, triangles,bound_L, bound_U)
        closest_point_set.append(current_closest_point)
        
    closest_point_set=np.array(closest_point_set)
    closest_point_set=np.squeeze(closest_point_set)
    
    return closest_point_set
'''
simple sphere method for searching nearest points in the surface:
    Input:
        tip_positons: the original positions of points to be searched with
        
        vertices: An Nx3 array, where each row is a vertice XYZ-coordinate
    
        triangle: An Nx3 array, where each row corresponding to the index of three
        vertices in variable 'vertices'
        
        centroid_path,radius_path,closest_point_path: the path to save centroid path,
        radius_path and closest_point_path computed in function
        
    Output:
       closest_point_set: the found point positions in surface 
'''
def step_2_simple_search_sphere(point_set, vertices, triangles,centroid_path,radius_path,closest_point_path):
    closest_point_set=[]
    centroids,radiuses = PAF3.bounding_sphere(vertices, triangles)
    np.save(centroid_path,centroids)
    np.save(radius_path,radiuses)
    for i in range(point_set.shape[0]):
        current_point=point_set[i,:]
        current_closest_point,_,_=PAF3.simple_search_sphere(current_point, vertices, triangles,centroids,radiuses)
        closest_point_set.append(current_closest_point)
        
    closest_point_set=np.array(closest_point_set)
    closest_point_set=np.squeeze(closest_point_set)
    np.save(closest_point_path,closest_point_set)
    return closest_point_set
'''
KDTree method for searching nearest points in the surface:
    Input:
        tip_positons: the original positions of points to be searched with
        
        vertices: An Nx3 array, where each row is a vertice XYZ-coordinate
    
        triangle: An Nx3 array, where each row corresponding to the index of three
        vertices in variable 'vertices'
        
    Output:
       closest_point_set: the found point positions in surface 
'''
def step_2_kd_tree_search(point_set, vertices, triangles):
    closest_point_set=[]
    centroids,radiuses = PAF3.bounding_sphere(vertices, triangles)
    kdtree=spatial.KDTree(data=centroids)
    distance, index = kdtree.query(point_set)
    for i in range(point_set.shape[0]):
        current_point=point_set[i,:]
        current_triangle_index=triangles[int(index[i]),:]
        current_vertice=np.zeros((3,3))
        current_vertice[0,:]=vertices[int(current_triangle_index[0]),:]
        current_vertice[1,:]=vertices[int(current_triangle_index[1]),:]
        current_vertice[2,:]=vertices[int(current_triangle_index[2]),:]
        current_closest_point,_=PAF3.simple_search_kdtree(current_point, current_vertice)
        closest_point_set.append(current_closest_point)
        
    closest_point_set=np.array(closest_point_set)
    closest_point_set=np.squeeze(closest_point_set)
    return closest_point_set
'''
output_generate function save the result as requested in PA3
'''
def output_generate(tip_positions,tip_positions_transformed,save_path,title):
    distance=np.zeros((tip_positions.shape[0]))
    f=open(save_path,'w')
    f.write(title)
    f.write('\n')
    for i in range(tip_positions.shape[0]):
        distance[i]=np.linalg.norm(tip_positions[i,:]-tip_positions_transformed[i,:])
        for j in range(3):
            f.write(str(format(tip_positions[i,j],'.2f'))+' ')
        for j in range(3):
            f.write(str(format(tip_positions_transformed[i,j],'.2f'))+' ')
        f.write(str(format(distance[i],'.2f'))+'\n')
        
    return distance
        
    