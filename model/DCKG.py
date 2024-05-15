import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from torch_scatter.utils import broadcast
from collections import OrderedDict

class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_items, n_attributes, triplet_attention, use_gate, without_any_gate):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_attributes = n_attributes
        self.triplet_attention = triplet_attention
        self.use_gate = use_gate
        self.without_any_gate = without_any_gate
        self.gate1 = nn.Linear(64, 64, bias=False)
        self.gate2 = nn.Linear(64, 64, bias=False)
        self.gate3 = nn.Linear(64, 64, bias=False)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def scatter_softmax(src, index, dim: int = -1, eps: float = 1e-12):
        if not torch.is_floating_point(src):
            raise ValueError('`scatter_softmax` can only be computed over tensors '
                             'with floating point data types.')

        index = broadcast(index, src, dim)

        max_value_per_index = scatter_max(src, index, dim=dim)[0]
        max_per_src_element = max_value_per_index.gather(dim, index)

        recentered_scores = src - max_per_src_element
        recentered_scores_exp = recentered_scores.exp()

        sum_per_index = scatter_sum(recentered_scores_exp, index, dim)
        normalizing_constants = sum_per_index.add_(eps).gather(dim, index)

        return recentered_scores_exp.div(normalizing_constants)

    @staticmethod
    def KG_forward(entity_emb, edge_index, edge_type, weight):
        torch.cuda.empty_cache()
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = weight[edge_type]  # (261494, 64)
        entity_emb_tail = entity_emb[tail]  # (464567, 64)
        neigh_relation_emb = entity_emb_tail * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        return entity_agg

    @staticmethod
    def UKG_forward(attribute_emb, edge_index, edge_type, weight):
        n_attributes = attribute_emb.shape[0]

        head, tail = edge_index
        edge_relation_emb = weight[edge_type]
        neigh_relation_emb = attribute_emb[tail] * edge_relation_emb  # [-1, channel]
        attribute_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_attributes, dim=0)

        return attribute_agg

    def user_gate(self, item_kg_agg, interact_mat, user_emb, weight, fast_weights, i):
        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]
        mat_val = interact_mat._values()

        item_neigh_emb = user_emb[mat_row] * weight[0]
        i_u_agg = scatter_mean(src=item_neigh_emb, index=mat_col, dim_size=self.n_items, dim=0)

        if fast_weights is None:
            gi = self.sigmoid(self.gate1(item_kg_agg) + self.gate2(i_u_agg))
        else:
            gate1_name = 'convs.{}.gate1.weight'.format(str(i))
            gate2_name = 'convs.{}.gate2.weight'.format(str(i))
            conv_w1 = fast_weights[gate1_name]
            conv_w2 = fast_weights[gate2_name]
            gi = self.sigmoid(F.linear(item_kg_agg, conv_w1) + F.linear(i_u_agg, conv_w2))

        item_emb_fusion = (gi * item_kg_agg) + ((1 - gi) * i_u_agg)
        user_item_mat = torch.sparse.FloatTensor(torch.cat([mat_row, mat_col]).view(2, -1),
                                                 torch.ones_like(mat_val),
                                                 size=[self.n_users, self.n_items])
        user_agg = torch.sparse.mm(user_item_mat, item_emb_fusion)
        return user_agg, item_emb_fusion

    def item_gate(self, user_ukg_agg, interact_mat, entity_emb, weight, fast_weights, i):
        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]
        mat_val = interact_mat._values()
        user_neigh_emb = entity_emb[mat_col] * weight[0]
        u_i_agg = scatter_mean(src=user_neigh_emb, index=mat_row, dim_size=self.n_users, dim=0)
        if fast_weights is None:
            hi = self.sigmoid(self.gate2(u_i_agg) + self.gate3(user_ukg_agg))
            # gi = self.sigmoid(self.gate1(item_kg_agg) + self.gate2(i_u_agg) + self.gate3(user_ukg_agg))
        else:
            gate2_name = 'convs.{}.gate2.weight'.format(str(i))
            gate3_name = 'convs.{}.gate3.weight'.format(str(i))
            conv_w2 = fast_weights[gate2_name]
            conv_w3 = fast_weights[gate3_name]

            hi = self.sigmoid(F.linear(u_i_agg, conv_w2) + F.linear(user_ukg_agg, conv_w3))

        user_emb_fusion = (hi * user_ukg_agg) + ((1 - hi) * u_i_agg)
        user_item_mat = torch.sparse.FloatTensor(torch.cat([mat_col, mat_row]).view(2, -1),
                                                 torch.ones_like(mat_val),
                                                 size=[self.n_items, self.n_users])
        item_agg = torch.sparse.mm(user_item_mat, user_emb_fusion)
        return user_emb_fusion, item_agg

    def without_gate(self, entity_agg, interact_mat, user_emb, weight, fast_weights, i):
        item_kg_agg = entity_agg[:self.n_items]
        att_kg_agg = entity_agg[self.n_items:]
        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]
        mat_val = interact_mat._values()

        item_neigh_emb = user_emb[mat_row] * weight[0]

        i_u_agg = scatter_mean(src=item_neigh_emb, index=mat_col, dim_size=self.n_items, dim=0)

        if fast_weights is None:
            gi = self.sigmoid(self.gate1(item_kg_agg) + self.gate2(i_u_agg))
        else:
            gate1_name = 'convs.{}.gate1.weight'.format(str(i))
            gate2_name = 'convs.{}.gate2.weight'.format(str(i))
            conv_w1 = fast_weights[gate1_name]
            conv_w2 = fast_weights[gate2_name]
            gi = self.sigmoid(F.linear(item_kg_agg, conv_w1) + F.linear(i_u_agg, conv_w2))

        item_emb_fusion = (gi * item_kg_agg) + ((1 - gi) * i_u_agg)
        user_item_mat = torch.sparse.FloatTensor(torch.cat([mat_row, mat_col]).view(2, -1),
                                                 torch.ones_like(mat_val),
                                                 size=[self.n_users, self.n_items])
        user_agg = torch.sparse.mm(user_item_mat, item_emb_fusion)
        entity_agg = torch.cat([item_emb_fusion, att_kg_agg])

        return entity_agg, user_agg

    def forward(self, entity_emb, user_emb, edge_index, edge_type, user_edge_index, user_edge_type,
                interact_mat, weight, fast_weights=None, i=0):

        """KG aggregate"""
        torch.cuda.empty_cache()
        entity_agg = self.KG_forward(entity_emb, edge_index, edge_type, weight)
        # print('self.n_users', self.n_users)
        # print('self.n_items', self.n_items)
        # print('entity_emb', entity_emb.shape)
        # print('interact_mat', interact_mat.shape)


        """user aggregate"""
        if self.use_gate:
            attribute_agg = self.UKG_forward(user_emb, user_edge_index, user_edge_type, weight)
            user_agg1, item_agg1 = self.user_gate(entity_agg[:self.n_items], interact_mat, user_emb, weight, fast_weights, i)
            user_agg2, item_agg2 = self.item_gate(attribute_agg[:self.n_users], interact_mat, entity_emb, weight, fast_weights, i)
            user_agg = torch.cat([user_agg2, attribute_agg[self.n_users:]])
            entity_agg = torch.cat([item_agg1, entity_agg[self.n_items:]])
        elif not self.without_any_gate:
            attribute_agg = user_emb[self.n_users:]
            entity_agg, user_agg = self.without_gate(entity_agg, interact_mat, user_emb, weight, fast_weights, i)
            user_agg = torch.cat([user_agg, attribute_agg])
        else:
            attribute_agg = user_emb[self.n_users:]
            mat_row = interact_mat._indices()[0, :]
            mat_col = interact_mat._indices()[1, :]
            mat_val = interact_mat._values()
            user_item_mat = torch.sparse.FloatTensor(torch.cat([mat_row, mat_col]).view(2, -1),
                                                     torch.ones_like(mat_val),
                                                     size=[self.n_users, entity_emb.shape[0]])
            user_agg = torch.sparse.mm(user_item_mat, entity_emb)
            user_agg = torch.cat([user_agg, attribute_agg])

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users, n_relations, n_items, n_attributes, use_gate, without_any_gate,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_attributes = n_attributes
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.triplet_attention = self.Consis_attention()

        weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_items=n_items, n_attributes=n_attributes,
                                         triplet_attention=self.triplet_attention, use_gate=use_gate, without_any_gate=without_any_gate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    @staticmethod
    def Consis_attention():
        # used in KCAN (CIKM 21), no parameter
        return nn.CosineSimilarity(dim=1, eps=1e-6)

    @staticmethod
    def _edge_sampling(edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    @staticmethod
    def _sparse_dropout(x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    @staticmethod
    def _edge_filter(edge_index, edge_type):
        # edge_num = [i for i, x in enumerate(edge_type) if x == 0 or x == 1]
        edge_num = (edge_type != 0).nonzero(as_tuple=True)[0]
        edge_index = edge_index[:, edge_num]
        edge_type = edge_type[edge_num]
        edge_num = (edge_type != 1).nonzero(as_tuple=True)[0]
        return edge_index[:, edge_num], edge_type[edge_num]

    def forward(self, user_emb, entity_emb, edge_index, edge_type, user_edge_index, user_edge_type,
                interact_mat, fast_weights=None, mess_dropout=True, node_dropout=True):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            # interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]

        # edge_attention_index, edge_attention_type = self._edge_filter(edge_index, edge_type)
        # back_attention_entity_emb = entity_emb
        # back_attention_user_emb = user_emb
        # back_attention_entity_emb, back_attention_user_emb = \
        #     self.convs[len(self.convs) - 1](back_attention_entity_emb, back_attention_user_emb,
        #                                     edge_attention_index, edge_attention_type,
        #                                     user_edge_index, user_edge_type,
        #                                     interact_mat,
        #                                     self.weight, fast_weights,
        #                                     i=len(self.convs) - 1)

        # if mess_dropout:
        #     back_attention_entity_emb = self.dropout(back_attention_entity_emb)
        #     back_attention_user_emb = self.dropout(back_attention_user_emb)
        # back_attention_entity_emb = F.normalize(back_attention_entity_emb)
        # back_attention_user_emb = F.normalize(back_attention_user_emb)
        # entity_res_emb = torch.add(entity_res_emb, back_attention_entity_emb)
        # user_res_emb = torch.add(user_res_emb, back_attention_user_emb)

        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb,
                                                 edge_index, edge_type,
                                                 user_edge_index, user_edge_type,
                                                 interact_mat,
                                                 self.weight, fast_weights, i=i)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
        """
        :todo 
        """
        return entity_res_emb, user_res_emb


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, ckg_graph, ukg_graph, user_pre_embed, item_pre_embed):
        """
        推荐器初始化函数
        :param data_config: 数据参数 n_为数量参数
        :args_config: 整体参数
        :graph: MultiDiGraph
        :user_pre_embed: 用户预嵌入
        :item_pre_embed: 物品预嵌入
        :return:
        """
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.n_attributes = data_config['n_attributes']
        self.user_pre_embed = user_pre_embed
        self.item_pre_embed = item_pre_embed

        # inner meta-learning update
        self.num_inner_update = args_config.num_inner_update
        self.meta_update_lr = args_config.meta_update_lr

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.use_gate = args_config.use_gate
        self.without_any_gate = args_config.without_any_gate
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.edge_index, self.edge_type = self._get_edges(ckg_graph)
        self.user_edge_index, self.user_edge_type = self._get_edges(ukg_graph)
        self._init_weight()
        self.gcn = self._init_model()
        self.interact_mat = None

    def _init_weight(self):
        self.all_embed = nn.init.xavier_uniform_(torch.empty(self.n_nodes, self.emb_size))

        if self.user_pre_embed is not None and self.item_pre_embed is not None:
            entity_emb = self.all_embed[(self.n_users + self.n_items): self.n_users + self.n_entities, :]
            attribute_emb = self.all_embed[self.n_users + self.n_entities:, :]
            self.all_embed = torch.cat([self.user_pre_embed, self.item_pre_embed, entity_emb, attribute_emb])

        self.all_embed = nn.Parameter(self.all_embed)  # （130319， 64）

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_items=self.n_items,
                         n_attributes=self.n_attributes,
                         use_gate=self.use_gate,
                         without_any_gate=self.without_any_gate,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def get_parameter(self):
        param_dict = dict()
        for name, para in self.gcn.named_parameters():
            if name.startswith('conv'):
                param_dict[name] = para

        return OrderedDict(param_dict)

    def forward_kg(self, h, r, pos_t, neg_t):
        entity_emb = self.all_embed[self.n_users: self.n_users + self.n_entities, :]
        h_emb = entity_emb[h]
        r_emb = entity_emb[r]
        pos_t_emb = entity_emb[pos_t]
        neg_t_emb = entity_emb[neg_t]

        r_t_pos = pos_t_emb * r_emb
        r_t_neg = neg_t_emb * r_emb

        pos_score = torch.sum(torch.pow(r_t_pos - h_emb, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_t_neg - h_emb, 2), dim=1)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        return kg_loss

    def forward_meta(self, support, query, fast_weights=None):
        user_s = support[0]
        pos_item_s = support[1]
        neg_item_s = support[2]
        user_q = query[0]
        pos_item_q = query[1]
        neg_item_q = query[2]

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users: self.n_users + self.n_entities, :]
        attribute_emb = self.all_embed[self.n_users + self.n_entities:, :]
        user_emb = torch.cat([user_emb, attribute_emb])

        if fast_weights is None:
            fast_weights = self.get_parameter()

        for i in range(self.num_inner_update):
            entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                    entity_emb,
                                                    self.edge_index,
                                                    self.edge_type,
                                                    self.user_edge_index,
                                                    self.user_edge_type,
                                                    self.interact_mat,
                                                    fast_weights=fast_weights,
                                                    mess_dropout=self.mess_dropout,
                                                    node_dropout=self.node_dropout)
            u_e = user_gcn_emb[user_s]
            pos_e, neg_e = entity_gcn_emb[pos_item_s], entity_gcn_emb[neg_item_s]
            loss, _, _ = self.create_bpr_loss(u_e, pos_e, neg_e)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=False)

            fast_weights = OrderedDict(
                (name, param - self.meta_update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                entity_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.user_edge_index,
                                                self.user_edge_type,
                                                self.interact_mat,
                                                fast_weights=fast_weights,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user_q]
        pos_e, neg_e = entity_gcn_emb[pos_item_q], entity_gcn_emb[neg_item_q]
        loss, _, _ = self.create_bpr_loss(u_e, pos_e, neg_e)

        return loss

    def forward(self, batch=None, is_apapt=False):
        if is_apapt:
            user = batch['users']
            pos_item = batch['pos_items']
            neg_item = batch['neg_items']
        else:
            user = batch[0]
            pos_item = batch[1]
            neg_item = batch[2]

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users: self.n_users + self.n_entities, :]
        attribute_emb = self.all_embed[self.n_users + self.n_entities:, :]
        user_emb = torch.cat([user_emb, attribute_emb])

        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                entity_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.user_edge_index,
                                                self.user_edge_type,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        loss, _, _ = self.create_bpr_loss(u_e, pos_e, neg_e)

        return loss

    def generate(self, adapt_fast_weight=None):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users: self.n_users + self.n_entities, :]
        attribute_emb = self.all_embed[self.n_users + self.n_entities:, :]
        user_emb = torch.cat([user_emb, attribute_emb])
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                entity_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.user_edge_index,
                                                self.user_edge_type,
                                                self.interact_mat,
                                                fast_weights=adapt_fast_weight,
                                                mess_dropout=False, node_dropout=False)

        return entity_gcn_emb, user_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
