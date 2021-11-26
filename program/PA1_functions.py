# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:33:00 2021

@author: Ding
"""

import numpy as np


def rotation(R,vector):
  rotation_matrix = R #Define the rotation matrix 
  determinant = np.linalg.det(rotation_matrix) #finding the determinanat 
  

  rotation_matrix_transpose = np.matrix.transpose(R) #Finding the transpose of rotation matrix 
  multiplication = np.matmul(rotation_matrix,rotation_matrix_transpose) #RR^T = I check orthogonality

  determinant = int(determinant)
  euclidian_distance = int(np.sum((multiplication-np.identity(3))**2))
  if ((determinant - 1)>1e-3):
     raise Exception("Not a Rotation matrix")
  if ((euclidian_distance)>1e-3):
    raise Exception("Not a Rotation matrix")

  new_vector = np.matmul(rotation_matrix,vector) 
  return new_vector

def frame_transformation(R,p,vector):
  point_transformed=rotation(R,vector)+p.reshape(3,1)
  return point_transformed
#This part is the SVD based points registeration function.

#The rotation function of processed points set A' B'
def points_registeration_rotation(A,B):
  H=np.matmul(np.matrix.transpose(A),B)
  U,S,V=np.linalg.svd(H,full_matrices=False)
  R=np.matmul(np.transpose(V),np.transpose(U))
  if np.abs(np.linalg.det(R) - 1) > 1e-3:
      raise Exception("Method fails!")
  
  return R

#SVD registeration function
def points_registeration(A,B):
  m_A=A-np.mean(A,axis=0)
  m_B=B-np.mean(B,axis=0)

  R=points_registeration_rotation(m_A,m_B)

  position=np.mean(B,axis=0).reshape(3,1)-np.matmul(R,np.mean(A,axis=0).reshape(3,1))


  #print('error', ((np.matmul(R,np.transpose(A))+position-np.transpose(B))**2).mean(axis=None))
  SVD_frame=np.hstack((R,position.reshape(3,1)))
  return SVD_frame

#This part is the quaternion based points registeration function.
def points_registeration_q1_rotation(A,B):
  H=np.matmul(np.matrix.transpose(A),B)
  HT=np.transpose(H)
  trace_H=np.trace(H)
  deltaT=np.array([H[1,2]-H[2,1],H[2,0]-H[0,2],H[0,1]-H[1,0]])
  delta=np.transpose(deltaT)
  G=np.zeros((4,4))
  G[0,0]=trace_H
  G[0,1:4]=deltaT
  G[1:4,0]=delta
  G[1:4,1:4]=H+HT-trace_H*np.eye(3)
  evals,evectors=np.linalg.eig(G)
  max_positions=np.argmax(evals)
  q = evectors[:,max_positions]
  Rotation_matrix=np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2,2*(q[1]*q[2]-q[0]*q[3]),2*(q[1]*q[3]+q[0]*q[2])],
               [2*(q[1]*q[2]+q[0]*q[3]),q[0]**2-q[1]**2+q[2]**2-q[3]**2,2*(q[2]*q[3]-q[0]*q[1])],
               [2*(q[1]*q[3]-q[0]*q[2]),2*(q[2]*q[3]+q[0]*q[1]),q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
  return Rotation_matrix

def points_registeration_q1(A,B):
  m_A=A-np.mean(A,axis=0)
  m_B=B-np.mean(B,axis=0)

  R=points_registeration_rotation(m_A,m_B)

  position=np.mean(B,axis=0).reshape(3,1)-np.matmul(R,np.mean(A,axis=0).reshape(3,1))
  #print('error',((np.matmul(R,np.transpose(A))+position-np.transpose(B))**2).mean(axis=None))
  Frame_Q=np.hstack((R,position.reshape(3,1)))
  return Frame_Q

#This part is for Q4.
#the inverse and composition operation used in Q4
def frame_inverse(R,p):
  R_inverse=np.linalg.inv(R)
  p_inverse=(-1)*np.matmul(R_inverse,p)
  Frame_inverse=np.hstack((R_inverse,p_inverse.reshape(3,1)))
  return Frame_inverse

def frame_composition(R1,p1,R2,p2):
  R_comp=np.matmul(R1,R2)
  p_comp=np.matmul(R1,p2)+p1
  frame_comp=np.hstack((R_comp,p_comp.reshape(3,1)))
  return frame_comp

#main part of Q4
def distortion_calibration_direct(d,D,a,A,c):

  F_D=points_registeration(d,D)
  
  F_A=points_registeration(a,A)
  
  F_D_inverse=frame_inverse(F_D[:,0:3],F_D[:,3])

  F_C=frame_composition(F_D_inverse[:,0:3],F_D_inverse[:,3].reshape(3,1),F_A[:,0:3],F_A[:,3].reshape(3,1))
#  print('F_C',F_C)
  C_expect=frame_transformation(F_C[:,0:3],F_C[:,3],c)

  return C_expect
