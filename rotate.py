from mpl_toolkits import mplot3d
import random
import numpy as np
import matplotlib.pyplot as plt
import pudb
import sys

def abs_angle(pos):
    if (pos == 0):
        return 0.
    elif (pos == 1):
        return np.pi
    elif (pos == 2):
        return np.pi / 2
    elif (pos == 3):
        return np.pi / 4
    elif (pos == 4):
        return 3 * np.pi / 4
    elif (pos == 5):
        return 3 * np.pi / 2
    elif (pos == 6):
        return 7 * np.pi / 4
    elif (pos == 7):
        return 5 * np.pi / 4

def get_angle_diff(init_pos, final_pos):
    # pu.db
    ang = - (abs_angle(int(final_pos)) - abs_angle(int(init_pos)))
    # if (ang < 0):
    #     ang = 2 * np.pi - ang
    return ang

def create_rot_matrix(init_pos, final_pos):
    ang = get_angle_diff(init_pos, final_pos)
    R = np.array([[np.cos(ang), -np.sin(ang), 0, 0],\
                  [np.sin(ang), np.cos(ang), 0, 0],\
                  [0, 0, 1, 0],\
                  [0, 0, 0, 1]])

    return R

def view_option(curr):
    print("""
        0|front 
        1|rear
        2|left
        3|left front
        4|left rear
        5|right
        6|right front
        7|right rear
        8|quit
        """)
    print("\nCurrent choice: "+str(curr))
    choice = input("Your choice: ")
    if (int(choice) >= 8 or int(choice) < 0):
        sys.exit(0)
    return choice

count = 0
curr_view = 0
pts = [(1.5, 1.5, 0.), (3.5, 1.5, 0.), (1.5, 3.5, 0.), (3.5, 3.5, 0.),\
                       (3.5, 1.5, 1.), (1.5, 3.5, 1.), (3.5, 3.5, 1.),\
                                       (1.5, 3.5, 2.), (3.5, 3.5, 2.),\
                                                      (3.5, 3.5, 3.)]

while True:
    count += 1
    fig = plt.figure(num='view '+str(curr_view))
    ax = plt.axes(projection="3d")


    pts_x = [i[0] for i in pts]
    pts_y = [i[1] for i in pts]
    pts_z = [i[2] for i in pts]

    # x_pos = [1.5, 3.5, 3.5, 1.5]
    # y_pos = [1.5, 1.5, 3.5, 3.5]
    # x_pos.extend(x_pos)
    # x_pos.extend(x_pos)
    # y_pos.extend(y_pos)
    # y_pos.extend(y_pos)
    # z_pos = [0., 0., 0., 0.]
    # z_pos.extend([1., 1., 1., 1.])
    # z_pos.extend([2., 2., 2., 2.])
    # # x_size = np.ones(num_bars)* 0.3
    # # y_size = np.ones(num_bars)* 0.3
    # # z_size = [1., 2., 3., 4.]

    ax.scatter3D(pts_x, pts_y, pts_z, color='b')
    plt.show(block=False)

    choice = view_option(curr_view)
    R = create_rot_matrix(curr_view, choice)
    for i in range(len(pts)):
        x = pts[i][0]
        y = pts[i][1]
        z = pts[i][2]
        x_new, y_new, z_new, _ = np.matmul(R, [[x], [y], [z], [1.]])
        x_new, y_new, z_new = x_new[0], y_new[0], z_new[0]
        pts[i] = (x_new, y_new, z_new)
    
    curr_view = choice