# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 14:11:12 2021

@author: Ding
"""
import numpy as np
import pandas as pd
import PA1_functions as PAF1
import re
def body_surface_loader(path):
    mesh_file=pd.read_csv(path)
    N_vertices=int(list(mesh_file)[0])
    N_triangle=int(mesh_file.iloc[N_vertices,0])
    vertices=np.zeros((N_vertices,3))
    for n_vertices in range(N_vertices):
        current_coordinates=mesh_file.iloc[n_vertices,0]
        current_coordinates=current_coordinates.split()
        vertices[n_vertices,0]=float(current_coordinates[0])
        vertices[n_vertices,1]=float(current_coordinates[1])
        vertices[n_vertices,2]=float(current_coordinates[2])
    
    triangles=np.zeros((N_triangle,3))
    for n_triangle in range(N_triangle):
        current_coordinates=mesh_file.iloc[n_triangle+N_vertices+1,0]
        current_coordinates=current_coordinates.split()
        triangles[n_triangle,0]=float(current_coordinates[0])
        triangles[n_triangle,1]=float(current_coordinates[1])
        triangles[n_triangle,2]=float(current_coordinates[2])
        
    return vertices, triangles


'''
a=R^{3} corner 3x3 each row is a vertices,[p,q,r] respectively
'''
def FindClosestPoint(a,corner):
    
    A=np.vstack((corner[1,:]-corner[0,:],corner[2,:]-corner[0,:])).transpose()
    B=a-corner[0,:]
    Lambda,_,_,_=np.linalg.lstsq(A,B)
    nearest_point = corner[0,:] +Lambda[0]*(corner[1,:]-corner[0,:])+Lambda[1]*(corner[2,:]-corner[0,:])

    if (Lambda[0]>=0) & (Lambda[1]>=0) & ((Lambda[0] + Lambda[1])<= 1) :
        nearest_point = corner[0,:] +Lambda[0]*(corner[1,:]-corner[0,:])+Lambda[1]*(corner[2,:]-corner[0,:])
        
    elif (Lambda[1]<0):
        ratio= (np.inner(nearest_point-corner[0,:],corner[1,:]-corner[0,:]))/(np.inner(corner[1,:]-corner[0,:],corner[1,:]-corner[0,:]))
        ratio=max(0,min(ratio,1))
        nearest_point=corner[0,:]+ratio*(corner[1,:]-corner[0,:])
    
    elif(Lambda[0]<0):
        ratio= (np.inner(nearest_point-corner[2,:],corner[0,:]-corner[2,:]))/(np.inner(corner[0,:]-corner[2,:],corner[0,:]-corner[2,:]))
        ratio=max(0,min(ratio,1))
        nearest_point=corner[2,:]+ratio*(corner[0,:]-corner[2,:])

    elif((Lambda[0] + Lambda[1])> 1):
        ratio=(np.inner(nearest_point-corner[1,:],corner[2,:]-corner[1,:]))/(np.inner(corner[2,:]-corner[1,:],corner[2,:]-corner[1,:]))
        ratio=max(0,min(ratio,1))
        nearest_point=corner[1,:]+ratio*(corner[2,:]-corner[1,:])
    
    return nearest_point    
'''
point R^3 point 
'''
def simple_search(point, vertices, triangles,bound_L, bound_U):
    bound=1e6

    for n_triangle in range(triangles.shape[0]):
        current_vertice = np.zeros((3,3))
        current_vertice[0,:]= vertices[int(triangles[n_triangle,0]),:]
        current_vertice[1,:]= vertices[int(triangles[n_triangle,1]),:]
        current_vertice[2,:]= vertices[int(triangles[n_triangle,2]),:]
        
        condition1=((bound_L[n_triangle,0]-bound)<=point[0]) & ((bound_U[n_triangle,0]+bound)>=point[0])
        condition2=((bound_L[n_triangle,1]-bound)<=point[1]) & ((bound_U[n_triangle,1]+bound)>=point[1])
        condition3=((bound_L[n_triangle,2]-bound)<=point[2]) & ((bound_U[n_triangle,2]+bound)>=point[2])
       
        if  condition1 & condition2 & condition3 :
            nearest_point=FindClosestPoint(point,current_vertice)
            
            distance = np.linalg.norm(nearest_point-point)
            
            if distance < bound:
                current_projection = nearest_point
                bound = distance
                minimal_index=n_triangle
    return current_projection,minimal_index, bound

def bound_fix(vertices, triangles):
    
    bound_L=np.zeros((triangles.shape[0],3))
    bound_U=np.zeros((triangles.shape[0],3))

    for n_triangle in range(triangles.shape[0]):
        current_vertice = np.zeros((3,3))
        current_vertice[0,:]= vertices[int(triangles[n_triangle,0]),:]
        current_vertice[1,:]= vertices[int(triangles[n_triangle,1]),:]
        current_vertice[2,:]= vertices[int(triangles[n_triangle,2]),:]
        
        bound_L[n_triangle,0]=np.min(current_vertice[:,0])        
        bound_L[n_triangle,1]=np.min(current_vertice[:,1])
        bound_L[n_triangle,2]=np.min(current_vertice[:,2])

        bound_U[n_triangle,0]=np.max(current_vertice[:,0])        
        bound_U[n_triangle,1]=np.max(current_vertice[:,1])
        bound_U[n_triangle,2]=np.max(current_vertice[:,2])
        
    return bound_L, bound_U

def bounding_sphere_compute(corner):
    length=np.zeros(3)
    length[0]=np.linalg.norm(corner[1,:]-corner[2,:])
    length[1]=np.linalg.norm(corner[0,:]-corner[2,:])
    length[2]=np.linalg.norm(corner[1,:]-corner[0,:])
    
    longest_length_index=np.argmax(length)
    AB=[]
    for i in range(3):
        if i != longest_length_index:
            AB.append(corner[i,:])
        else:
            C=corner[i,:]
    A=AB[0]
    B=AB[1]
    
    Q=(A+B)/2
    
    if np.inner(C-Q,C-Q) <= np.inner(A-Q,A-Q):
        centroid = Q
    else:
        F=(A+B)/2
        U=A-F
        V=C-F
        D=np.cross(np.cross(U,V),U)
        
        gamma = (np.inner(V,V)-np.inner(U,U))/(2*np.inner(D,V-U))
        
        if gamma <= 0:
            Lambda = 0
        else:
            Lambda = gamma
        
        centroid = F + Lambda * D
        
    radius = np.linalg.norm(centroid-A)
    return centroid, radius    

def bounding_sphere(vertices, triangles):
    total_centroid=np.zeros((triangles.shape[0],3))
    total_radius=np.zeros((triangles.shape[0]))
    for n_triangle in range(triangles.shape[0]):
        current_vertice = np.zeros((3,3))
        current_vertice[0,:]= vertices[int(triangles[n_triangle,0]),:]
        current_vertice[1,:]= vertices[int(triangles[n_triangle,1]),:]
        current_vertice[2,:]= vertices[int(triangles[n_triangle,2]),:]
        
        total_centroid[n_triangle,:],total_radius[n_triangle]=bounding_sphere_compute(current_vertice)
    
    
    return total_centroid, total_radius

def simple_search_sphere(point, vertices, triangles,centroids,radiuses ):
    bound=1e6

    for n_triangle in range(triangles.shape[0]):
        current_vertice = np.zeros((3,3))
        current_vertice[0,:]= vertices[int(triangles[n_triangle,0]),:]
        current_vertice[1,:]= vertices[int(triangles[n_triangle,1]),:]
        current_vertice[2,:]= vertices[int(triangles[n_triangle,2]),:]
        if np.linalg.norm(centroids[n_triangle,:]-point)-radiuses[n_triangle] <= bound:
            nearest_point=FindClosestPoint(point,current_vertice)
            
            distance = np.linalg.norm(nearest_point-point)
            
            if distance < bound:
                current_projection = nearest_point
                bound = distance
                minimal_index=n_triangle
    return current_projection,minimal_index, bound

        
'''
ICP is the iterated closest point algorithm we construct.
Input: 
    tip_positions:A tip positions in Nx3 array
    vertices: vertices coordinates generated by PA3F.body_surface_loader function
    triangles: vertices from PA3F.body_surface_loader function, the triangle vertices indices
    F_init: 4x3 array where the first 3 cols corresponding to R and the last col is position
    max_iteration: maximal iteration for ICP
    termination_threshold: the threshold to dertermin if early stopping
    termination_iteration:continues iteration, if met the iteration is early stopped
    gamma:params to check if early stooping
    
Output:
    frame: learned frame
    mean_ICP_distance: distance between the transformed point and its closest point
    ICP_tansformed: transformed point set
    ICP_nearest_point: corresponding closest point set
    error_matrix: mean error for each iteration
'''    
def ICP(point_set, vertices, triangles, F_init, max_iteration, termination_threshold = 1, termination_iteration = 5, gamma=0.95):
    outlier_threshold=1e3
    current_Frame=F_init
    bound_L, bound_U=bound_fix(vertices, triangles)
    error_matrix=np.zeros(max_iteration)
    continue_iteration=0
    for opt_iteration in range(max_iteration):
################### MATCHING #######################    
      print(opt_iteration)
      current_point_transformed = PAF1.frame_transformation(current_Frame[:,0:3],current_Frame[:,3],point_set.T)
      current_point_transformed = current_point_transformed.T
      A=[]
      B=[]
      
      for n_point in range(current_point_transformed.shape[0]):
          current_projection,minimal_index, distance=simple_search(current_point_transformed[n_point,:], vertices, triangles,bound_L, bound_U)
          
          if distance < outlier_threshold:
              A.append(point_set[n_point,:])
              B.append(current_projection)

              
      A=np.array(A)
      B=np.array(B)
#################### UPDATE #######################      
      current_Frame=PAF1.points_registeration(A,B)
      updated_point_coordinate =PAF1.frame_transformation(current_Frame[:,0:3],current_Frame[:,3],A.T)
      updated_point_coordinate=updated_point_coordinate.T
      error=0
      for n_point in range(A.shape[0]):          
          error = error +np.linalg.norm(updated_point_coordinate[n_point,:]-B[n_point,:])
          
      mean_error = error / A.shape[0]
      
      if outlier_threshold > 3*mean_error:
        outlier_threshold=  3*mean_error
      
      error_matrix[opt_iteration]=mean_error
################ CHECKING TERMINATION ##################
      if opt_iteration >= 1:
          ratio = error_matrix[opt_iteration]/error_matrix[opt_iteration-1]
          if ratio>=gamma and ratio<=1:
             continue_iteration=continue_iteration+1
          
          if ((continue_iteration >= termination_iteration) and (mean_error < termination_threshold)) or (opt_iteration >=max_iteration-1):
              return_transformed_coordinate = PAF1.frame_transformation(current_Frame[:,0:3],current_Frame[:,3],point_set.T)
              return_transformed_coordinate=return_transformed_coordinate.T
              return_closest_coordinate=[]
              for n_point in range(current_point_transformed.shape[0]):
                  current_projection,_, _=simple_search(return_transformed_coordinate[n_point,:], vertices, triangles,bound_L, bound_U)
                  return_closest_coordinate.append(current_projection)
              return_closest_coordinate=np.array(return_closest_coordinate)
              return current_Frame,mean_error,return_transformed_coordinate,return_closest_coordinate,error_matrix

def PA4_result_comparison(answer_read_path, ICP_tansformed,ICP_nearest_point):
    f=open(answer_read_path)
    line = f.readline()
    tranformed_standard = np.zeros((ICP_tansformed.shape[0],3))
    closest_standard = np.zeros((ICP_tansformed.shape[0],3))
    error_transformed=np.zeros(ICP_tansformed.shape[0])
    error_ICP = np.zeros(ICP_tansformed.shape[0])
    for i in range(ICP_tansformed.shape[0]):
        line = f.readline()
        split_line = re.findall(r"\D?\d+.?\d*",line)
        tranformed_standard[i,:]=np.array(split_line[0:3])
        closest_standard[i,:] = np.array(split_line[3:6])
        error_transformed[i] = np.linalg.norm(tranformed_standard[i,:]-ICP_tansformed[i,:])
        error_ICP[i] = np.linalg.norm(ICP_nearest_point[i,:]-closest_standard[i,:])
    f.close()
    error_transformed_mean = np.mean(error_transformed)
    error_ICP_mean = np.mean(error_ICP)
    return error_transformed_mean,error_ICP_mean
    
    
    
def simple_search_kdtree(point, vertice):
    nearest_point=FindClosestPoint(point,vertice)
    distance = np.linalg.norm(nearest_point-point)
    
    return nearest_point, distance