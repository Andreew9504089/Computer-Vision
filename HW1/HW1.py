import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
import scipy
from scipy.integrate import dblquad

image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image

def ComputeNormalMap(img_src, light_src):
    KdN = np.linalg.inv(np.transpose(light_src)@light_src)@np.transpose(light_src)@img_src
    N = np.transpose(KdN)
    for row in range(N.shape[0]):
        N[row] /= (np.linalg.norm(N[row]) + 0.0001)
    
    return N

def ComputeDepthMap(normal_map, mask):
    normal_map = np.copy(np.reshape(normal_map, (image_row, image_col, 3)))
    [Y, X] = np.where(mask > 0)
    coord = np.dstack((Y,X))[0]
    S = len(coord)
    M = np.zeros((2*S,S))
    V = np.zeros((2*S,1))

    idx_map = np.zeros((mask.shape), dtype=np.int64)
    for idx in range(S):
        idx_map[coord[idx][0], coord[idx][1]] = int(idx)

    for idx in range(S):
        x = coord[idx][1]
        y = coord[idx][0]
        n = normal_map[y,x]

        if mask[y,x+1] > 0 and mask[y-1,x] > 0:
            M[idx, idx] = -1
            tmp = idx_map[y,x+1]
            M[idx, tmp] = 1
            V[idx] = -n[0]/n[2]

            M[idx+S, idx] = -1
            tmp = idx_map[y-1,x]
            M[idx+S, tmp] = 1
            V[idx+S] = -n[1]/n[2]

        # (x+1, y) is not valid
        elif mask[y-1,x] > 0:
            if mask[y, x-1] > 0:
                M[idx, idx] = -1
                tmp = idx_map[y,x-1]
                M[idx, tmp] = 1
                V[idx] = n[0]/n[2]

            M[idx+S, idx] = -1
            tmp = idx_map[y-1,x]
            M[idx+S, tmp] = 1
            V[idx+S] = -n[1]/n[2]

        # (x, y+1) is not valid
        elif mask[y, x+1] > 0:
            if mask[y+1,x] > 0:
                M[idx+S, idx] = -1
                tmp = idx_map[y+1,x]
                M[idx+S, tmp] = 1
                V[idx+S] = n[1]/n[2]

            M[idx, idx] = -1
            tmp = idx_map[y,x+1]
            M[idx, tmp] = 1
            V[idx] = -n[0]/n[2]

        # both is not valid
        else:
            if mask[y+1,x] > 0:
                M[idx+S, idx] = -1
                tmp = idx_map[y+1,x]
                M[idx+S, tmp] = 1
                V[idx+S] = n[1]/n[2]

            if mask[y, x-1] > 0:
                M[idx, idx] = -1
                tmp = idx_map[y,x-1]
                M[idx, tmp] = 1
                V[idx] = n[0]/n[2]
            
    M = csr_matrix(M)
    Z = scipy.sparse.linalg.lsqr(M,V)[0]

    z = np.zeros((image_row,image_col))

    idx = 0
    for p in coord:
        z[p[0], p[1]] = Z[idx]
        idx += 1

    return z

def CreateMask(normal_map):
    map_1D = np.reshape(np.sum(normal_map, axis = 1), ((image_row,image_col))).copy()
    mask = np.where(map_1D != 0, 1, 0)
    return mask

if __name__ == '__main__':
    
    names = ['bunny', 'star', 'venus']
    for name in names:
        print(name)
        save_path = 'result'+name+'_result.ply'
        bmps = None
        lights = None

        # Read light source data
        datas = pd.read_csv('test/'+name+'/LightSource.txt', sep=" ", header=None)
        for data in datas[1]:
            light_vec = np.asarray(eval(data))
            unit_light = light_vec/np.sqrt(np.sum(light_vec**2))
            if lights is None:
                lights = unit_light
            else:
                lights = np.vstack([lights, unit_light])

        # Read Bitmap image
        for i in range(6):
            if bmps is None:
                bmps = np.reshape(read_bmp('test/'+name+'/pic'+str(i+1)+'.bmp'), -1)
            else:
                bmps = np.vstack([bmps, np.reshape(read_bmp('test/'+name+'/pic'+str(i+1)+'.bmp'), -1)])


        N = ComputeNormalMap(bmps, lights)
        normal_visualization(N)

        M = CreateMask(N)
        mask_visualization(M)

        Z = ComputeDepthMap(N, M)
        depth_visualization(Z)

        # save_ply(Z,save_path)
        # show_ply(save_path)

    # showing the windows of all visualization function
    plt.show()