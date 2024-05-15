import math
import random
import torch
import os
import numpy as np
from time import time
from prettytable import PrettyTable

from utility.parser_DCKG import parse_args
from utility.data_loader import load_data
from utility.data_loader import reload_ckg_graph, reload_ukg_graph, build_ckg_graph
from model.DCKG import Recommender
from utility.evaluate import test
from utility.helper import early_stopping
from utility.scheduler import Scheduler
from collections import OrderedDict
from tqdm import tqdm
from sklearn.cluster import KMeans
# from kmeans_pytorch import kmeans
from utility.torch_kmeans import kmeans
from time import strftime
import datetime

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
sample_num = 10


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).to(device)
    return feed_dict


def get_max_entity(args):
    """
    return: dict{org_id: remap_id}   type: {str: str}
    """
    path = args.data_path + '{}/'.format(args.dataset) + 'entity_list.txt'
    lines = open(path, 'r').readlines()
    max_entity = 0
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        max_entity = max(max_entity, int(tmp[-1]))
    return max_entity + 1


def get_max_attribute(args):
    """
    return: dict{org_id: remap_id}   type: {str: str}
    """
    path = args.data_path + '{}/'.format(args.dataset) + 'attribute_list.txt'
    lines = open(path, 'r').readlines()
    max_attribute = 0
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        max_attribute = max(max_attribute, int(tmp[-2]))
    return max_attribute + 1


def get_feed_dict_meta(support_user_set):
    """

    :param support_user_set:
    :return: support_meta_set: n个用户 每个用户dict包含 用户id 10个正例物品 10个负例物品
    """
    support_meta_set = []
    for key, val in support_user_set.items():
        feed_dict = []
        user = [int(key)] * sample_num
        if len(val) != sample_num:
            pos_item = np.random.choice(list(val), sample_num, replace=True)
        else:
            pos_item = val

        neg_item = []
        while True:
            tmp = np.random.randint(low=0, high=n_items, size=1)[0]
            if tmp not in val:
                neg_item.append(tmp)
            if len(neg_item) == sample_num:
                break
        feed_dict.append(np.array(user))
        feed_dict.append(np.array(list(pos_item)))
        feed_dict.append(np.array(neg_item))
        support_meta_set.append(feed_dict)

    return np.array(support_meta_set)  # [n_user, 3, 10]


def get_feed_kg(kg_graph):
    triplet_num = len(kg_graph)
    pos_hrt_id = np.random.randint(low=0, high=triplet_num, size=args.batch_size * sample_num)
    pos_hrt = kg_graph[pos_hrt_id]
    neg_t = np.random.randint(low=0, high=n_entities, size=args.batch_size * sample_num)

    return torch.LongTensor(pos_hrt[:, 0]).to(device), torch.LongTensor(pos_hrt[:, 1]).to(device), torch.LongTensor(
        pos_hrt[:, 2]).to(device), torch.LongTensor(neg_t).to(device)


def convert_to_sparse_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape).to(device)


def get_net_parameter_dict(params):
    param_dict = dict()
    indexes = []
    for i, (name, param) in enumerate(params):
        if param.requires_grad:
            param_dict[name] = param.to(device)
            indexes.append(i)

    return param_dict, indexes


def update_moving_avg(mavg, reward, count):
    return mavg + (reward.item() - mavg) / (count + 1)


def renew_item_cluster(item_emb, can_triplets_np):
    best_k = 2000
    label_type_dict = {'last-fm': 10, 'yelp2018': 46, 'amazon-book': 40}
    label_type = label_type_dict[args.dataset]
    print('start get labels')
    # labels = KMeans(n_clusters=best_k, max_iter=50, tol=1e-3).fit_predict(item_emb)
    labels, centroids = kmeans(X=item_emb, num_clusters=best_k, tol=1e-3, device=device, mat_iter=50)
    print('get labels')

    new_edge = []
    max_entity = get_max_entity(args)
    for i, label in enumerate(labels):
        new_edge.append([i, label_type, label + max_entity])
    new_edge = np.array(new_edge)
    can_triplets_np = np.concatenate([can_triplets_np, new_edge], axis=0)
    can_triplets_np = np.unique(can_triplets_np, axis=0)
    print('start reload ckg garph')
    ckg_graph = reload_ckg_graph(args, can_triplets_np)
    print('end reload ckg garph')
    return ckg_graph


def renew_user_cluster(user_emb):
    best_k = 2000
    label_type_dict = {'last-fm': 10, 'yelp2018': 46, 'amazon-book': 40}
    label_type = label_type_dict[args.dataset]
    print('start get labels')
    # labels = KMeans(n_clusters=best_k, max_iter=50, tol=1e-3).fit_predict(item_emb)
    labels, centroids = kmeans(X=user_emb, num_clusters=best_k, tol=1e-3, device=device, mat_iter=50)
    print('get labels')

    new_edge = []
    max_attribute = get_max_attribute(args)
    for i, label in enumerate(labels):
        new_edge.append([i, label_type, label + max_attribute])
    new_edge = np.array(new_edge)
    print('start reload ukg garph')
    ukg_graph = reload_ukg_graph(args, new_edge)
    print('end reload ukg garph')
    return ukg_graph


if __name__ == '__main__':
    """fix the random seed"""
    # 设置随机种子
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    # args.user_clustered = False
    print("args.user_clustered", args.user_clustered)
    print("args.use_network_model", args.use_network_model)
    # cuda = True gpu_id = 0
    # args.cuda = False
    # args.gpu_id = 0
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    """build dataset"""
    # cold_scenario = user_item_cold  help="[user_cold, item_cold, user_item_cold, warm_up]"
    cold_scenario = args.cold_scenario  # the cold scenario adapted
    train_cf, test_cf, user_dict, n_params, ckg_graph, ukg_graph, mat_list = load_data(args, 'meta_training')
    adj_mat_list, mean_mat_list = mat_list
    cold_train_cf, cold_test_cf, cold_user_dict, cold_n_params, cold_ckg_graph, cold_ukg_graph, cold_mat_list = \
        load_data(args, cold_scenario)
    cold_adj_mat_list, cold_mean_mat_list = cold_mat_list

    kg_graph = np.array(list(ckg_graph.edges))  # [-1, 3]
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    # test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))
    cold_train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in cold_train_cf], np.int32))
    # cold_test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in cold_test_cf], np.int32))

    """use pretrain data"""
    if args.use_pretrain:
        pre_path = args.data_path + 'pretrain/{}/mf.npz'.format(args.dataset)
        pre_data = np.load(pre_path)
        user_pre_embed = torch.tensor(pre_data['user_embed'])
        item_pre_embed = torch.tensor(pre_data['item_embed'])
    else:
        user_pre_embed = None
        item_pre_embed = None

    """init model"""
    model = Recommender(n_params, args, ckg_graph, ukg_graph, user_pre_embed, item_pre_embed).to(device)
    names_weights_copy, indexes = get_net_parameter_dict(model.named_parameters())
    # print(names_weights_copy)
    scheduler = Scheduler(len(names_weights_copy), grad_indexes=indexes).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_update_lr)
    scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=args.scheduler_lr)

    """prepare feed data"""
    support_meta_set = get_feed_dict_meta(user_dict['train_user_set'])
    query_meta_set = get_feed_dict_meta(user_dict['test_user_set'])
    # shuffle
    index = np.arange(len(support_meta_set))
    np.random.shuffle(index)
    support_meta_set = support_meta_set[index]
    query_meta_set = query_meta_set[index]

    # support_cold_set = get_feed_dict_meta(cold_user_dict['train_user_set'])

    if args.use_meta_model:
        if args.clustered:
            model.load_state_dict(torch.load('./model_para/meta_model_clustered_{}.ckpt'.format(args.dataset)))
            # model.load_state_dict(torch.load('./model_para/meta_model_{}.ckpt'.format(args.dataset)))
        else:
            # model.load_state_dict(torch.load('./model_para/meta_model_clustered_{}.ckpt'.format(args.dataset)))
            model.load_state_dict(torch.load('./model_para/meta_model_{}.ckpt'.format(args.dataset)))
    else:
        print("start meta training ...")
        """meta training"""
        # meta-training ui_interaction
        interact_mat = convert_to_sparse_tensor(mean_mat_list)
        model.interact_mat = interact_mat
        moving_avg_reward = 0

        model.train()  # 启用 batch normalization 和 dropout
        iter_num = math.ceil(len(support_meta_set) / args.batch_size)
        train_s_t = time()
        for s in tqdm(range(iter_num)):
            batch_support = torch.LongTensor(support_meta_set[s * args.batch_size:(s + 1) * args.batch_size]).to(device)
            batch_query = torch.LongTensor(query_meta_set[s * args.batch_size:(s + 1) * args.batch_size]).to(device)

            pt = int(s / iter_num * 100)
            if len(batch_support) > args.meta_batch_size:
                task_losses, weight_meta_batch = scheduler.get_weight(batch_support, batch_query, model, pt)
                torch.cuda.empty_cache()
                task_prob = torch.softmax(weight_meta_batch.reshape(-1), dim=-1)
                selected_tasks_idx = scheduler.sample_task(task_prob, args.meta_batch_size)
                batch_support = batch_support[selected_tasks_idx]
                batch_query = batch_query[selected_tasks_idx]

            selected_losses = scheduler.compute_loss(batch_support, batch_query, model)
            meta_batch_loss = torch.mean(selected_losses)

            """KG loss"""
            h, r, pos_t, neg_t = get_feed_kg(kg_graph)
            kg_loss = model.forward_kg(h, r, pos_t, neg_t)
            batch_loss = kg_loss + meta_batch_loss

            """update scheduler"""
            loss_scheduler = 0
            for idx in selected_tasks_idx:
                loss_scheduler += scheduler.m.log_prob(idx.cuda())
                # loss_scheduler += scheduler.m.log_prob(idx)
            reward = meta_batch_loss
            loss_scheduler *= (reward - moving_avg_reward)
            moving_avg_reward = update_moving_avg(moving_avg_reward, reward, s)

            scheduler_optimizer.zero_grad()
            loss_scheduler.backward(retain_graph=True)
            scheduler_optimizer.step()

            """update network"""
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            if args.save and (s % 100 == 0 or s == iter_num - 1 or s == iter_num - 2 or s == iter_num - 3):
                if args.clustered:
                    torch.save(model.state_dict(), args.out_dir + 'meta_model_clustered_' + args.dataset + '.ckpt')
                else:
                    torch.save(model.state_dict(), args.out_dir + 'meta_model_' + args.dataset + '.ckpt')

        train_e_t = time()
        print('meta_training_time: ', train_e_t - train_s_t)

    """fine tune"""
    # adaption ui_interaction
    cold_interact_mat = convert_to_sparse_tensor(cold_mean_mat_list)
    model.interact_mat = cold_interact_mat

    if args.user_clustered:
        clustered = '_user_clustered_dynamic'
    elif args.clustered:
        clustered = '_clustered_dynamic'
    else:
        clustered = ''
    if args.use_gate:
        use_gate = ''
    elif args.without_any_gate:
        use_gate = '_without_any_gate'
    else:
        use_gate = '_without_gate'
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    print("start fine tune...")
    start_epoch = 0
    print("use_network_model:", args.use_network_model)
    print("clustered:", clustered)
    print("use_gate:", use_gate)
    file_name = args.data_path + args.dataset + '/' + 'kg_final.txt'
    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    if args.use_network_model:
        if args.clustered:
            # model.load_state_dict(torch.load('./model_para/networks_model_clustered_{}.ckpt'.format(args.dataset)))
            model.load_state_dict(torch.load('./model_para/networks_model{0}{1}_'.format(clustered,
                                                                                         use_gate) + args.dataset + args.cold_scenario + '.ckpt'))
        else:
            model.load_state_dict(torch.load('./model_para/networks_model_{}.ckpt'.format(args.dataset)))
    # reset lr
    for g in optimizer.param_groups:
        g['lr'] = args.lr
    model.gcn.node_dropout_rate = args.node_dropout_rate
    for epoch in range(args.epoch):
        # shuffle training data
        index = np.arange(len(cold_train_cf))
        np.random.shuffle(index)
        cold_train_cf_pairs = cold_train_cf_pairs[index]

        model.train()
        loss = 0
        iter_num = math.ceil(len(cold_train_cf) / args.fine_tune_batch_size)
        train_s_t = time()
        for s in tqdm(range(iter_num)):
            batch = get_feed_dict(cold_train_cf_pairs,
                                  s * args.fine_tune_batch_size, (s + 1) * args.fine_tune_batch_size,
                                  cold_user_dict['train_user_set'])
            batch_loss = model(batch, is_apapt=True)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
        train_e_t = time()

        # new_edge_index = torch.zeros(2, model.n_items)
        # entity_num = {'last-fm': 106389}
        # # for i, label in enumerate(labels):
        # #     print(i, label)
        # edge_index = model.edge_index
        # edge_type = model.edge_type
        # edge_num = (edge_type == 10).nonzero(as_tuple=True)[0]
        # edge_index[1, edge_num] = labels + entity_num[args.dataset]
        # if epoch == 0 and args.clustered == True:
        #     args.clustered = False
        #     ckg_graph = build_ckg_graph(args)
        if epoch % 50 == 0 and args.clustered and epoch != 0:

            ckg_graph = renew_item_cluster(model.all_embed[model.n_users: model.n_users + model.n_items, :].detach(), can_triplets_np)
            model.edge_index, model.edge_type = model._get_edges(ckg_graph)
            print('end get edges')
            ckg_graph = []
            torch.cuda.empty_cache()

        if epoch % 50 == 0 and epoch != 0 and args.user_clustered:
            ukg_graph = renew_user_cluster(model.all_embed[: model.n_users, :].detach())
            model.user_edge_index, model.user_edge_type = model._get_edges(ukg_graph)
            ukg_graph = []
            print('end get edges')
            torch.cuda.empty_cache()

        if epoch % 5 == 0 or epoch == 1:
            """testing"""
            model.eval()
            torch.cuda.empty_cache()
            test_s_t = time()
            with torch.no_grad():
                ret = test(model, cold_user_dict, cold_n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg"]
            train_res.add_row(
                [epoch + start_epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg'], ])
            print(train_res)
            print('start save', epoch + start_epoch)

            if args.save:
                torch.save(model.state_dict(),
                           args.out_dir + 'networks_model{0}{1}_'.format(clustered,
                                                                         use_gate) + args.dataset + args.cold_scenario + '.ckpt')

            # if args.save:
            #     if args.clustered:
            #         if args.user_clustered:
            #             torch.save(model.state_dict(),
            #                        args.out_dir + 'networks_model_user_clustered_dynamic_' + args.dataset + args.cold_scenario + '.ckpt')
            #         else:
            #             torch.save(model.state_dict(),
            #                        args.out_dir + 'networks_model_clustered_dynamic_' + args.dataset + args.cold_scenario +'.ckpt')
            #     else:
            #         torch.save(model.state_dict(), args.out_dir + 'networks_model_' + args.dataset + '.ckpt')
            f = open(
                './result/{}_{}_bt{}_lr{}_metaLr{}{}{}.txt'.format(args.dataset, cold_scenario,
                                                                   args.fine_tune_batch_size,
                                                                   args.lr, args.meta_update_lr, clustered, use_gate),
                'a+')
            f.write(str(train_res) + '\n')
            f.close()
            print('end save', epoch)
            # early stopping.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=20)
            if should_stop:
                break

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
    current_time = datetime.datetime.now()
    print("current_time:    " + str(current_time))
