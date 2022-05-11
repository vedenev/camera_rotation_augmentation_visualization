#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def deg_to_rad(a):
    return a * np.pi / 180

def imshow_bgr(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)

def int_round(inp):
    return int(np.round(inp))

def plot_pyramid(ax, v, facecolors, edgecolors, alpha):
    # https://stackoverflow.com/questions/39408794/python-3d-pyramid

    ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
    
    # generate list of sides' polygons of our pyramid
    verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
     [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]
    
    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, 
                                         facecolors=facecolors,
                                         linewidths=1,
                                         edgecolors=edgecolors,
                                         alpha=alpha))

def get_vectors_vertex(h, w, f):
    v = np.zeros((5, 3), np.float32)
    v[0: 4, 2] = f

    v[0, 0] = w / 2
    v[0, 1] = h / 2

    v[1, 0] = -w / 2
    v[1, 1] = h / 2

    v[2, 0] = -w / 2
    v[2, 1] = -h / 2

    v[3, 0] = w / 2
    v[3, 1] = -h / 2
    
    v[4, :] = 0

    return v

height = 1080
width = 1920
focal_dist = 1481
image = cv2.imread("images/frame_indoor_2022_05_04_horizontal.png")

height_2 = 500
width_2 = 1000
focal_dist_2 = 3000

a_x_deg = -25
a_y_deg = -12

a_x = deg_to_rad(a_x_deg)
a_y = deg_to_rad(a_y_deg)

focal_dist = np.float32(focal_dist)
focal_dist_2 = np.float32(focal_dist_2)
width_half = np.float32(width / 2)
height_half = np.float32(height / 2)
width_2_half = np.float32(width_2 / 2)
height_2_half = np.float32(height_2 / 2)

# for x is around y:
R_for_x = np.asarray([[np.cos(a_x), 0, -np.sin(a_x)],
                      [0, 1, 0],
                      [np.sin(a_x), 0, np.cos(a_x)]], np.float32)

# for y is around x:
R_for_y = np.asarray([[1, 0, 0],
                      [0, np.cos(a_y), -np.sin(a_y)],
                      [0, np.sin(a_y), np.cos(a_y)]], np.float32)

R = np.matmul(R_for_x, R_for_y)

x = np.arange(width_2, dtype=np.float32) - width_2_half
y = np.arange(height_2, dtype=np.float32) - height_2_half
X, Y = np.meshgrid(x, y)
Z = focal_dist_2 * np.ones_like(X)
# rotation from new coordinates to original, as cv2.remap requires
X_rotated = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z
Y_rotated = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z
Z_rotated = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z
X_projected = focal_dist * X_rotated / Z_rotated
Y_projected = focal_dist * Y_rotated / Z_rotated
map_x = X_projected + width_half
map_y = Y_projected + height_half
image_2 = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC)



points_x = [map_x[0, 0], map_x[0, -1], map_x[-1, -1], map_x[-1, 0]]
points_y = [map_y[0, 0], map_y[0, -1], map_y[-1, -1], map_y[-1, 0]]

points_x.append(points_x[0])
points_y.append(points_y[0])

X_projected_limits = np.asarray([X_projected[0, 0], 
                                 X_projected[0, -1], 
                                 X_projected[-1, -1], 
                                 X_projected[-1, 0]])

Y_projected_limits = np.asarray([Y_projected[0, 0], 
                                 Y_projected[0, -1], 
                                 Y_projected[-1, -1], 
                                 Y_projected[-1, 0]])

Z_projected_limits = focal_dist * np.ones_like(Y_projected_limits)

verts_proj = np.stack((X_projected_limits, 
                       Y_projected_limits,
                       Z_projected_limits))

verts_proj = [verts_proj.T.tolist()]


pyramid_vertecies = get_vectors_vertex(height, width, focal_dist)

pyramid_vertecies_2 = get_vectors_vertex(height_2, width_2, focal_dist_2)
pyramid_vertecies_2 = (np.matmul(R, pyramid_vertecies_2.T)).T

fig = plt.figure()
fig.set_size_inches(10, 3)

ax = fig.add_subplot(1, 3, 1, projection='3d')


plot_pyramid(ax, pyramid_vertecies, 'cyan', 'r', 0.25)


plot_pyramid(ax, pyramid_vertecies_2, 'magenta', 'r', 0.25)

coll_proj = Poly3DCollection(verts_proj,
                             facecolors='blue',
                             linewidths=1,
                             edgecolors='r',
                             alpha=0.25)
ax.add_collection3d(coll_proj)


ax.invert_yaxis()
ax.auto_scale_xyz([-1500, 1500], [-1500, 1500], [0, 3000])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=46, azim=74)
plt.title("camera models")








plt.subplot(1, 3, 2)
imshow_bgr(image)
plt.plot(points_x, points_y, 'r-')
margin = 110
plt.xlim([-margin, image.shape[1] + margin])
plt.ylim([image.shape[0] + margin, -margin])
plt.title("original image")

plt.subplot(1, 3, 3)
imshow_bgr(image_2)
plt.title("augmented image")


plt.show()


