import numpy as np


def get_distance(Data):
    row=Data.shape[0]
    dis=np.zeros([row,row])
    for i in range(row-1):
        for j in range(i+1,row):
            v_i = Data[i, :]    # 数据点i
            v_j = Data[j, :]    # 数据点j
            dis[i, j] = np.sqrt(np.dot((v_i - v_j), (v_i - v_j)))    # sqrt开根号，dot点乘
    dis=dis+dis.T
    return dis

def get_dc(dis,percent):
    n=dis.shape[0]
    dis_arr=np.reshape(dis,n*n)
    position=int(n*(n-1)*percent+n-1)
    dc=np.sort(dis_arr)[position]
    return dc

def get_local_density(dis,dc):
    n=dis.shape[0]
    local_density=np.zeros(n)
    for i in range(n):
        local_density[i]=np.where(dis[i][:]<dc)[0].shape[0]-1
    return local_density

def get_deltas(distance_matrix, rhos):
    n = np.shape(distance_matrix)[0]
    deltas = np.zeros(n)
    nearest_neighbor = np.zeros(n)
    rhos_index = np.argsort(-rhos)  # 得到密度ρ从大到小的排序的索引

    for i, index in enumerate(rhos_index):
        # i是序号，index是rhos_index[i]，是第i大的ρ的索引，这里第0大是最大的。
        # index是点的索引，rhos[index]和deltas[index]是第index个点的ρ和δ
        if i == 0:
            continue
        higher_rhos_index = rhos_index[:i]  # 对于i，比这个点密度更大的点的索引号是rhos_index[:i]
        deltas[index] = np.min(distance_matrix[index, higher_rhos_index])  # 在index这一行比它ρ大的点中选最小的距离
        nearest_neighbors_index = np.argmin(distance_matrix[index, higher_rhos_index])  # distance_matrix第index行里面，higher_rhos_index这些列，中的值里面最小的，的索引（在higher_rhos_index中的索引）
        nearest_neighbor[index] = higher_rhos_index[nearest_neighbors_index].astype(int)
    deltas[rhos_index[0]] = np.max(deltas)
    return deltas, nearest_neighbor

def find_k_centers(rhos, deltas, k):

    rho_and_delta = rhos * deltas
    centers = np.argsort(-rho_and_delta)
    return centers[:k]

def density_peal_cluster(rhos, centers, nearest_neighbor):
    k = np.shape(centers)[0]
    if k == 0:
        print("Can't find any center")
        return
    n = np.shape(rhos)[0]
    labels = -1 * np.ones(n).astype(int)

    # 给刚刚找出来的簇心编号0， 1， 2， 3 ......
    for i, center in enumerate(centers):
        labels[center] = i

    # 再将每个点编上与其最近的高密度点相同的编号
    rhos_index = np.argsort(-rhos)
    for i, index in enumerate(rhos_index):
        if labels[index] == -1:
            labels[index] = labels[int(nearest_neighbor[index])]
    return labels