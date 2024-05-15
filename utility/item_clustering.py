from sklearn.cluster import KMeans
import torch
from utility.parser_DCKG import parse_args
import matplotlib.pyplot as plt
from utility.torch_kmeans import kmeans
# from kmeans_pytorch import kmeans
import numpy as np
from tqdm import tqdm
import shutil

def read_entity_list(path):
    """
    return: dict{org_id: remap_id}   type: {str: str}
    """
    path = path + 'entity_list.txt'
    lines = open(path, 'r').readlines()
    item_dict = dict()
    max_entity = 0
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        item_dict[tmp[0]] = tmp[-1]
        max_entity = max(max_entity, int(tmp[-1]))
        # item_dict[tmp[0]] = str(idx - 1)

    return item_dict, max_entity


if __name__ == '__main__':

    args = parse_args()
    device = torch.device("cuda:0")
    pre_path = '../' + args.data_path + 'pretrain/{}/mf.npz'.format(args.dataset)
    pre_data = np.load(pre_path)
    user_pre_embed = torch.tensor(pre_data['user_embed'])
    item_pre_embed = torch.tensor(pre_data['item_embed'])
    path = '../' + args.data_path + '{}/'.format(args.dataset)
    entity_dict, max_entity = read_entity_list(path)
    max_entity = max_entity + 1
    print(max_entity)
    # 读取数据

    # 创建 k-means 模型并拟合数据
    # kmeans = KMeans(n_clusters=30, random_state=0).fit(item_pre_embed)
    # k_range = list(range(100, 10000, 100))
    # sse = []
    # for k in tqdm(k_range):
    #     km = KMeans(n_clusters=k)
    #     km.fit(item_pre_embed)
    #     sse.append(km.inertia_)

    # 绘制SSE与k值的图表
    # plt.plot(k_range, sse)
    # plt.xlabel('K')
    # plt.ylabel('SSE')
    # plt.title('Elbow Method for Optimal K')
    # plt.show()

    best_k = 2000
    label_type = 39
    cluster_ids_x, cluster_centers = kmeans(X=item_pre_embed, num_clusters=best_k, device=device, tol=1e-3, mat_iter=100)
    # item_pre_embed = torch.tensor(item_pre_embed).to(device)
    # mykmeans.fit(item_pre_embed)

    # 获取聚类中心
    centroids = cluster_centers
    #
    # # 获取每个样本的标签
    labels = cluster_ids_x.numpy()
    #
    # 输出结果
    with open('../' + args.data_path + '{}/item_centroids.txt'.format(args.dataset), 'w') as f:
        for i,centroid in enumerate(centroids):
            f.writelines(str(centroid) + ' ' +str(i+max_entity)+'\n')
    # 定义源文件和目标文件的路径
    src_file = '../' + args.data_path + '{}/kg_final.txt'.format(args.dataset)
    dst_file = '../' + args.data_path + '{}/clustered_kg_final.txt'.format(args.dataset)

    # 复制文件
    shutil.copy(src_file, dst_file)
    with open('../' + args.data_path + '{}/clustered_kg_final.txt'.format(args.dataset), 'a') as f:
        for i,label in enumerate(labels):
            f.writelines(str(i) + ' {} '.format(str(label_type)) +str(int(label)+max_entity)+'\n')
    print('聚类中心:', centroids)
    print('标签:', labels)



