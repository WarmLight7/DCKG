import pandas as pd
import gzip
import numpy as np
import json
import tqdm
import random
import collections
import time
from tqdm import tqdm

random.seed(2020)


def extract_ui_rating_amazon(path):
    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    temp_user_dict = collections.defaultdict(int)
    temp_item_dict = collections.defaultdict(int)
    for ori_id, remap_id in user_dict.items():
        temp_user_dict[ori_id] = int(remap_id) + 1
    for ori_id, remap_id in item_dict.items():
        temp_item_dict[ori_id] = int(remap_id) + 1

    rating_list_ui = collections.defaultdict(list)
    g = gzip.open(path + 'rawdata/reviews_Books_5.json.gz', 'r')
    for idx, l in enumerate(g):
        l = eval(l)
        if temp_user_dict[l['reviewerID']] != 0 and temp_item_dict[l['asin']] != 0:
            u_id = int(user_dict[l['reviewerID']])
            i_id = int(item_dict[l['asin']])
            rating = int(l['overall'])
            rating_list_ui[u_id].append([i_id, rating])

        if idx % 100000 == 0:
            print('idx: ', idx)

    with open(path + '/test_scenario/rating_list_ui.json', 'w') as f:
        json.dump(rating_list_ui, f)


def extract_ui_rating_yelp(path):
    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    temp_user_dict = collections.defaultdict(int)
    temp_item_dict = collections.defaultdict(int)
    for ori_id, remap_id in user_dict.items():
        temp_user_dict[ori_id] = int(remap_id) + 1
    for ori_id, remap_id in item_dict.items():
        temp_item_dict[ori_id] = int(remap_id) + 1

    rating_list_ui = collections.defaultdict(list)
    num = 0
    with open(path + 'rawdata/yelp_academic_dataset_review.json', 'r') as f:
        for line in f:
            tmp = json.loads(line)
            if temp_user_dict[tmp['user_id']] != 0 and temp_item_dict[tmp['business_id']] != 0:
                num += 1
                u_id = int(user_dict[tmp['user_id']])
                i_id = int(item_dict[tmp['business_id']])
                rating = int(tmp['stars'])
                rating_list_ui[u_id].append([i_id, rating])

                if num % 100000 == 0:
                    print(num)

    with open(path + '/test_scenario/rating_list_ui.json', 'w') as f:
        json.dump(rating_list_ui, f)


def first_reach_amazon(path):
    """
    the first timestamp user and item reached
    {user: unixtime}->{str: int}
    """
    first_reach_user = {}
    first_reach_item = {}
    user_item_interaction = {}
    item_user_interaction = {}

    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')
    g = gzip.open(path + 'rawdata/reviews_Books_5.json.gz', 'r')

    for idx, l in (enumerate(g)):
        l = eval(l)

        if l['reviewerID'] in list(user_dict.keys()):
            if l['reviewerID'] not in list(first_reach_user.keys()):
                first_reach_user[l['reviewerID']] = l['unixReviewTime']
            else:
                if l['unixReviewTime'] < first_reach_user[l['reviewerID']]:
                    first_reach_user[l['reviewerID']] = l['unixReviewTime']

        if l['asin'] in list(item_dict.keys()):
            if l['asin'] not in list(first_reach_item.keys()):
                first_reach_item[l['asin']] = l['unixReviewTime']
            else:
                if l['unixReviewTime'] < first_reach_item[l['asin']]:
                    first_reach_item[l['asin']] = l['unixReviewTime']
        if idx % 100000 == 0:
            print(l['reviewerID'], l['asin'], l['unixReviewTime'])
            print('idx: ', idx)

    print('user_dict: ', len(list(user_dict.keys())))
    print('first_reach_user: ', len(list(first_reach_user.keys())))
    print('item_dict: ', len(list(item_dict.keys())))
    print('first_reach_item: ', len(list(first_reach_item.keys())))

    with open(path + 'first_reach_user.json', 'w') as f:
        json.dump(first_reach_user, f)
    with open(path + 'first_reach_item.json', 'w') as f:
        json.dump(first_reach_item, f)


def first_reach_lfm(path):
    """
    the first timestamp user and item reached
    {user: unixtime}->{str: int}
    """
    first_reach_user = {}
    first_reach_item = {}

    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    user_lines = open(path + 'rawdata/LFM-1b_users.txt', 'r').readlines()

    u_unixtime = {}
    for idx, line in enumerate(user_lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        u_unixtime[tmp[0]] = tmp[-1]
    for key, val in user_dict.items():
        first_reach_user[key] = int(u_unixtime[key])

    lfm_LEs = open(path + 'rawdata/LFM-1b_LEs.txt', 'r').readlines()
    track_time = collections.defaultdict(int)
    for idx, line in enumerate(lfm_LEs):
        l = line.strip()
        tmp = l.split()
        track_id = int(tmp[-2])
        unixtime = int(tmp[-1])

        if track_time[track_id] == 0:
            track_time[track_id] = unixtime
        elif unixtime < track_time[track_id]:
            track_time[track_id] = unixtime

        if idx % 10000000 == 0:
            # print('track: ', type(track_id), track_id, 'unixtime: ', type(unixtime), unixtime)
            print('idx: ', idx)

    del lfm_LEs

    item_ids = [int(it) for it in list(item_dict.keys())]
    loss_item = 0
    for it_id in item_ids:
        if track_time[it_id] == 0:
            loss_item += 1
        first_reach_item[it_id] = track_time[it_id]
    print('loss_item_num: ', loss_item)

    print('user_dict: ', len(list(user_dict.keys())))
    print('first_reach_user: ', len(list(first_reach_user.keys())))
    print('item_dict: ', len(list(item_dict.keys())))
    print('first_reach_item: ', len(list(first_reach_item.keys())))

    with open(path + 'first_reach_user.json', 'w') as f:
        json.dump(first_reach_user, f)
    with open(path + 'first_reach_item.json', 'w') as f:
        json.dump(first_reach_item, f)


def first_reach_yelp(path):
    """
    the first timestamp user and item reached
    {user/item: unixtime}->{str: int}
    """
    first_reach_user = {}
    first_reach_item = {}

    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    print('start collect user_time...')
    user_time = collections.defaultdict(int)
    with open(path + 'rawdata/yelp_academic_dataset_user.json', 'r', encoding='utf-8') as f:
        for line in f:
            tmp = json.loads(line)
            # print(tmp)
            # print(type(tmp))
            # print(type(tmp['user_id']), tmp['user_id'], type(tmp['yelping_since']), tmp['yelping_since'])
            user_id = tmp['user_id']
            unixtime = time.strptime(tmp['yelping_since'], "%Y-%m-%d %H:%M:%S")
            unixtime = int(time.mktime(unixtime))
            user_time[user_id] = unixtime

    no_user = 0
    for key, val in user_dict.items():
        if user_time[key] == 0:
            no_user += 1
        else:
            first_reach_user[key] = user_time[key]
    print('no_user_num: ', no_user)

    print('start collect business_time...')
    business_time = collections.defaultdict(int)
    with open(path + 'rawdata/yelp_academic_dataset_review.json', 'r') as f:
        for line in f:
            tmp = json.loads(line)
            # print(tmp)
            # print(type(tmp))
            # print(type(tmp['business_id']),tmp['business_id'], type(tmp['date']),tmp['date'])
            business_id = tmp['business_id']
            unixtime = time.strptime(tmp['date'], "%Y-%m-%d %H:%M:%S")
            unixtime = int(time.mktime(unixtime))

            if business_time[business_id] == 0:
                business_time[business_id] = unixtime
            elif unixtime < business_time[business_id]:
                business_time[business_id] = unixtime

    no_item = 0
    for key, val in item_dict.items():
        if business_time[key] == 0:
            no_item += 1
        else:
            first_reach_item[key] = business_time[key]
    print('no_item_num: ', no_item)

    print('user_dict: ', len(list(user_dict.keys())))
    print('first_reach_user: ', len(list(first_reach_user.keys())))
    print('item_dict: ', len(list(item_dict.keys())))
    print('first_reach_item: ', len(list(first_reach_item.keys())))

    with open(path + 'first_reach_user.json', 'w') as f:
        json.dump(first_reach_user, f)
    with open(path + 'first_reach_item.json', 'w') as f:
        json.dump(first_reach_item, f)


def read_user_list(path):
    """
    return: dict{org_id: remap_id} type: {str: str}
    """
    lines = open(path, 'r').readlines()
    user_dict = dict()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        user_dict[tmp[0]] = tmp[1]

    return user_dict


def read_item_list(path):
    """
    return: dict{org_id: remap_id}   type: {str: str}
    """
    lines = open(path, 'r').readlines()
    item_dict = dict()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        item_dict[tmp[0]] = tmp[1]
        # item_dict[tmp[0]] = str(idx - 1)

    return item_dict


def read_attribute_list(path):
    """
    return: dict{org_id: remap_id}   type: {str: str}
    """
    lines = open(path, 'r').readlines()
    attribute_dict = dict()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        attribute_dict[tmp[0] + tmp[2]] = tmp[1]
        # item_dict[tmp[0]] = str(idx - 1)
    return attribute_dict


def merge_train_vali_test(path):
    """
    return: entire {users: items}
    """
    user_dict = dict()
    lines_train = open(path + 'train.txt', 'r').readlines()
    lines_vali = open(path + 'valid1.txt', 'r').readlines()
    lines_test = open(path + 'test.txt', 'r').readlines()
    for l_train, l_vali, l_test in zip(lines_train, lines_vali, lines_test):
        tmp_train = l_train.strip()
        tmp_vali = l_vali.strip()
        tmp_test = l_test.strip()
        inter_train = [int(i) for i in tmp_train.split()]
        inter_vali = [int(i) for i in tmp_vali.split()]
        inter_test = [int(i) for i in tmp_test.split()]

        user_id_train, item_ids_train = inter_train[0], inter_train[1:]
        user_id_vali, item_ids_vali = inter_vali[0], inter_vali[1:]
        user_id_test, item_ids_test = inter_test[0], inter_test[1:]
        item_ids_train = set(item_ids_train)
        item_ids_vali = set(item_ids_vali)
        item_ids_test = set(item_ids_test)

        item_ids_merge = item_ids_train | item_ids_vali | item_ids_test
        user_dict[user_id_train] = list(item_ids_merge)

    with open(path + 'user_items_all.json', 'w') as f:
        json.dump(user_dict, f)


def construct_test_scenario(path):
    """
    return: test_scenario
    """
    with open(path + 'first_reach_user.json', 'r') as f:
        first_reach_user = json.load(f)
    with open(path + 'first_reach_item.json', 'r') as f:
        first_reach_item = json.load(f)

    # sorted by timestamp
    user_timestamp = sorted(first_reach_user.items(), key=lambda x: x[1])
    item_timestamp = sorted(first_reach_item.items(), key=lambda x: x[1])

    print(len(user_timestamp), len(item_timestamp))
    print('type_user: ', type(user_timestamp[0][0]), 'type_timestamp: ', type(user_timestamp[0][1]))

    # (org_id, timestamp)  exist:new == 8:2
    new_user = user_timestamp[int(0.8 * len(user_timestamp)):]
    exist_user = user_timestamp[:int(0.8 * len(user_timestamp))]
    new_item = item_timestamp[int(0.8 * len(item_timestamp)):]
    exist_item = item_timestamp[:int(0.8 * len(item_timestamp))]

    print(len(new_user), len(exist_user), len(new_item), len(exist_item))

    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    # get remap_id of user or item
    new_user = [int(user_dict[t[0]]) for t in new_user]
    exist_user = [int(user_dict[t[0]]) for t in exist_user]
    new_item = [int(item_dict[t[0]]) for t in new_item]
    exist_item = [int(item_dict[t[0]]) for t in exist_item]

    print(new_user[:5])
    print(new_item[:5])

    # construct the test_scenario
    meta_training = dict()
    warm_up = dict()
    user_cold = dict()
    item_cold = dict()
    user_item_cold = dict()

    with open(path + 'user_items_all.json', 'r') as f:
        user_item_all = json.load(f)
    for key, value in tqdm(user_item_all.items()):
        if int(key) in new_user:
            user_cold[int(key)] = list(set(value) & set(exist_item))
            user_item_cold[int(key)] = list(set(value) & set(new_item))
        elif int(key) in exist_user:
            item_cold[int(key)] = list(set(value) & set(new_item))
            meta_training[int(key)] = list(set(value) & set(exist_item))

    for i in tqdm(range(int(0.1 * len(exist_user)))):
        if len(meta_training.keys()) == 0:
            continue
        idx = random.sample(meta_training.keys(), 1)[0]
        print(idx, meta_training[idx])
        warm_up[idx] = meta_training[idx]
        del meta_training[idx]

    with open(path + 'test_scenario/' + 'meta_training.json', 'w') as f:
        json.dump(meta_training, f)
    print('meta_training.json complete')
    with open(path + 'test_scenario/' + 'warm_up.json', 'w') as f:
        json.dump(warm_up, f)
    print('warm_up.json complete')
    with open(path + 'test_scenario/' + 'user_cold.json', 'w') as f:
        json.dump(user_cold, f)
    print('user_cold.json complete')
    with open(path + 'test_scenario/' + 'item_cold.json', 'w') as f:
        json.dump(item_cold, f)
    print('item_cold.json complete')
    with open(path + 'test_scenario/' + 'user_item_cold.json', 'w') as f:
        json.dump(user_item_cold, f)
    print('user_item_cold.json complete')


def support_query_set(path):
    state = ['meta_training', 'warm_up', 'user_cold', 'item_cold', 'user_item_cold']
    path_test = path + 'test_scenario/'
    for s in state:
        path_json = path_test + s + '.json'
        with open(path_json, 'r') as f:
            scenario = json.load(f)
        support_txt = open(path_test + s + '_support.txt', mode='w')
        query_txt = open(path_test + s + '_query.txt', mode='w')

        for u, i in scenario.items():
            if 13 <= len(i) <= 100:
                random.shuffle(i)
                support = i[:-10]
                query = i[-10:]
                support_txt.write(u)
                query_txt.write(u)
                for s_one in support:
                    support_txt.write(' ' + str(s_one))
                for q_one in query:
                    query_txt.write(' ' + str(q_one))
                support_txt.write('\n')
                query_txt.write('\n')
        support_txt.close()
        query_txt.close()



def create_user_item_list(path):
    collect_user = False
    collect_item = False
    collect_friend = False
    collect_user_attribute = True
    collect_relation = False
    collect_kg = False
    if collect_user:
        print('start collect user_list...')
        user_list = collections.defaultdict(int)
        user_num = 0

        with open(path + 'user_list.txt', 'a', encoding='utf-8') as user_file:
            with open(path + 'yelp_academic_dataset_user.json', 'r', encoding='utf-8') as f:
                for line in f:
                    tmp = json.loads(line)
                    user_id = tmp['user_id']
                    if user_list.get(user_id):
                        continue

                    user_list[user_id] = user_num
                    user_file.writelines(user_id + ' ' + str(user_num) + '\n')
                    user_num = user_num + 1
    if collect_item:
        print('start collect item_list...')
        item_list = collections.defaultdict(int)
        item_num = 0
        with open(path + 'item_list.txt', 'a', encoding='utf-8') as item_file:
            with open(path + 'yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
                for line in f:
                    tmp = json.loads(line)
                    item_id = tmp['business_id']
                    if item_list.get(item_id):
                        continue

                    item_list[item_id] = item_num
                    item_file.writelines(item_id + ' ' + str(item_num) + '\n')
                    item_num = item_num + 1
    if collect_friend:
        print('start collect friend_list...')
        friend_num = 0
        user_dict = read_user_list(path + 'user_list.txt')
        with open(path + 'friend_list.txt', 'w', encoding='utf-8') as friend_file:
            with open(path + 'yelp_academic_dataset_user.json', 'r', encoding='utf-8') as f:
                for line in f:
                    tmp = json.loads(line)
                    # print(tmp)
                    user_id = tmp['user_id']
                    user_friends = tmp['friends']
                    user_friends = user_friends.split(', ')
                    for friend_id in user_friends:
                        if user_dict.get(friend_id) and user_dict.get(user_id):
                            friend_file.writelines(user_dict[user_id] + ' ' + user_dict[friend_id] + '\n')
                    friend_num = friend_num + 1
        print('total collect {0} friend'.format(friend_num))
    if collect_kg:
        print('start collect kg_final...')
        triplet_num = 0
        # lfm_LEs = open('./datasets/last-fm/rawdata/LFM-1b_LEs.txt', 'r').readlines()
        # track_time = collections.defaultdict(int)
        # for idx, line in enumerate(lfm_LEs):
        #     l = line.strip()
        #     tmp = l.split()
        #     print(l)
        # quit(0)
        user_dict = read_user_list(path + 'user_list.txt')
        item_dict = read_item_list(path + 'item_list.txt')
        with open(path + 'kg_final.txt', 'w', encoding='utf-8') as kg_file:
            with open(path + 'yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
                for line in f:
                    tmp = json.loads(line)
                    # print(tmp)
                    user_id = tmp['user_id']
                    item_id = tmp['business_id']
                    user_id = user_dict[user_id]
                    item_id = item_dict[item_id]
                    kg_file.writelines(user_id + " 0 " + item_id + '\n')
                    triplet_num = triplet_num + 1
            with open(path + 'yelp_academic_dataset_tip.json', 'r', encoding='utf-8') as f:
                for line in f:
                    tmp = json.loads(line)
                    # print(tmp)
                    user_id = tmp['user_id']
                    item_id = tmp['business_id']
                    user_id = user_dict[user_id]
                    item_id = item_dict[item_id]
                    kg_file.writelines(user_id + " 1 " + item_id + '\n')
                    triplet_num = triplet_num + 1

        print('total collect {0} triplet'.format(triplet_num))
    quit(0)
    if collect_kg:
        print('start collect kg_final...')


def get_playcount(playcount):
    playcount_level = len(playcount)
    return str(playcount_level)


def create_attribute_lfm(path):
    print('start collect attribute_list')
    with open(path + 'attribute_list.txt', 'w', encoding='utf-8') as file:
        file.writelines('attribute_id\tattribute_value\tattribute_type\n')
        user_lines = open(path + 'rawdata/LFM-1b_users.txt', 'r').readlines()
        attribute_set = set()
        attribute_num = 0
        print('start collect country ' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        for idx, line in enumerate(user_lines):
            if idx == 0:
                continue
            line = line.strip()
            line = line.split('\t')
            if line[1] == '':
                continue
            attribute = (line[1], 'country')
            if attribute in attribute_set:
                continue
            attribute_set.add(attribute)
            file.writelines(line[1] + '\t' + str(attribute_num) + '\t' + 'country' + '\n')
            attribute_num = attribute_num + 1
        print('total collect attribute_num ' + str(attribute_num))

        print('start collect age ' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        for idx, line in enumerate(user_lines):
            if idx == 0:
                continue
            line = line.strip()
            line = line.split('\t')
            if line[2] == '':
                continue
            attribute = (line[2], 'age')
            if attribute in attribute_set:
                continue
            attribute_set.add(attribute)
            file.writelines(line[2] + '\t' + str(attribute_num) + '\t' + 'age' + '\n')
            attribute_num = attribute_num + 1
        print('total collect attribute_num ' + str(attribute_num))

        print('start collect gender ' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        attribute = ('n', 'gender')
        attribute_set.add(attribute)
        file.writelines(line[3] + '\t' + str(attribute_num) + '\t' + 'gender' + '\n')
        attribute_num = attribute_num + 1
        attribute = ('m', 'gender')
        attribute_set.add(attribute)
        file.writelines(line[3] + '\t' + str(attribute_num) + '\t' + 'gender' + '\n')
        attribute_num = attribute_num + 1
        print('total collect attribute_num ' + str(attribute_num))

        print('start collect playcount ' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        for i in range(0, 10):
            playcount_level = str(i)
            attribute = (playcount_level, 'playcount')
            attribute_set.add(attribute)
            file.writelines(playcount_level + '\t' + str(attribute_num) + '\t' + 'playcount' + '\n')
            attribute_num = attribute_num + 1
        print('total collect attribute_num ' + str(attribute_num))

        print('start collect novelty_artist ' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        for i in range(0, 10):
            num_level = str(i)
            attribute = (num_level, 'novelty_artist')
            attribute_set.add(attribute)
            file.writelines(num_level + '\t' + str(attribute_num) + '\t' + 'novelty_artist' + '\n')
            attribute_num = attribute_num + 1
        print('total collect attribute_num ' + str(attribute_num))

        print('start collect mainstreaminess ' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        for i in range(0, 10):
            num_level = str(i)
            attribute = (num_level, 'mainstreaminess')
            attribute_set.add(attribute)
            file.writelines(num_level + '\t' + str(attribute_num) + '\t' + 'mainstreaminess' + '\n')
            attribute_num = attribute_num + 1
        print('total collect attribute_num ' + str(attribute_num))

        print('start collect listeningevents ' + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        for i in range(0, 10):
            num_level = str(i)
            attribute = (num_level, 'cnt_listeningevents')
            attribute_set.add(attribute)
            file.writelines(num_level + '\t' + str(attribute_num) + '\t' + 'cnt_listeningevents' + '\n')
            attribute_num = attribute_num + 1
        print('total collect attribute_num ' + str(attribute_num))


def create_attribute_kg_lfm(path):
    print('start collect attribute_kg')
    user_dict = read_user_list(path + 'user_list.txt')
    attribute_dict = read_attribute_list(path + 'attribute_list.txt')
    with open(path + 'attribute_kg.txt', 'w', encoding='utf-8') as file:
        user_lines = open(path + 'rawdata/LFM-1b_users.txt', 'r').readlines()
        edge_num = 0
        print('start collect country ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        triplets_set = set()
        for idx, line in enumerate(user_lines):
            if idx == 0:
                continue
            line = line.strip()
            line = line.split('\t')
            if line[1] == '':
                continue
            if not user_dict.get(line[0]) or not attribute_dict.get(line[1]+'country'):
                continue
            user = user_dict[line[0]]
            edge = 10
            attribute = attribute_dict[line[1]+'country']
            edge = str(edge)
            triplet = (user, edge, attribute)
            if triplet in triplets_set:
                continue
            triplets_set.add(triplet)
            file.writelines(user + '\t' + edge + '\t' + attribute + '\n')
            edge_num = edge_num + 1
        print('total collect edge_num ' + str(edge_num))

        print('start collect age ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        for idx, line in enumerate(user_lines):
            if idx == 0:
                continue
            line = line.strip()
            line = line.split('\t')
            if line[2] == '':
                continue
            if not user_dict.get(line[0]) or not attribute_dict.get(line[2]+'age'):
                continue
            user = user_dict[line[0]]
            edge = 11
            attribute = attribute_dict[line[2]+'age']
            edge = str(edge)
            triplet = (user, edge, attribute)
            if triplet in triplets_set:
                continue
            triplets_set.add(triplet)
            file.writelines(user + '\t' + edge + '\t' + attribute + '\n')
            edge_num = edge_num + 1
        print('total collect edge_num ' + str(edge_num))

        print('start collect gender ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        for idx, line in enumerate(user_lines):
            if idx == 0:
                continue
            line = line.strip()
            line = line.split('\t')
            if line[3] == '':
                continue
            if not user_dict.get(line[0]) or not attribute_dict.get(line[3]+'gender'):
                continue
            user = user_dict[line[0]]
            edge = 12
            attribute = attribute_dict[line[3] + 'gender']
            edge = str(edge)
            triplet = (user, edge, attribute)
            if triplet in triplets_set:
                continue
            triplets_set.add(triplet)
            file.writelines(user + '\t' + edge + '\t' + attribute + '\n')
            edge_num = edge_num + 1
        print('total collect edge_num ' + str(edge_num))

        print('start collect playcount ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        for idx, line in enumerate(user_lines):
            if idx == 0:
                continue
            line = line.strip()
            line = line.split('\t')
            if line[4] == '':
                continue
            attribute_num = get_playcount(line[4])
            if not user_dict.get(line[0]) or not attribute_dict.get(attribute_num+'playcount'):
                continue
            user = user_dict[line[0]]
            edge = 13
            attribute = attribute_dict[attribute_num + 'playcount']
            edge = str(edge)
            triplet = (user, edge, attribute)
            if triplet in triplets_set:
                continue
            triplets_set.add(triplet)
            file.writelines(user + '\t' + edge + '\t' + attribute + '\n')
            edge_num = edge_num + 1
        print('total collect edge_num ' + str(edge_num))

        user_lines = open(path + 'rawdata/LFM-1b_users_additional.txt', 'r').readlines()
        for idx, line in enumerate(user_lines):
            if idx == 0:
                continue
            line = line.strip()
            line = line.split('\t')
            if line[1] == '':
                continue
            if len(line[1]) < 3:
                attribute_num = str(0)
            else:
                attribute_num = line[1][2]
            if not user_dict.get(line[0]) or not attribute_dict.get(attribute_num+'novelty_artist'):
                continue
            user = user_dict[line[0]]
            edge = 14
            attribute = attribute_dict[attribute_num + 'novelty_artist']
            edge = str(edge)
            triplet = (user, edge, attribute)
            if triplet in triplets_set:
                continue
            triplets_set.add(triplet)
            file.writelines(user + '\t' + edge + '\t' + attribute + '\n')
            edge_num = edge_num + 1
        print('total collect edge_num ' + str(edge_num))

        for idx, line in enumerate(user_lines):
            if idx == 0:
                continue
            line = line.strip()
            line = line.split('\t')
            if line[4] == '':
                continue
            if len(line[4]) < 3:
                attribute_num = str(0)
            else:
                attribute_num = line[4][2]
            if not user_dict.get(line[0]) or not attribute_dict.get(attribute_num+'mainstreaminess'):
                continue
            user = user_dict[line[0]]
            edge = 15
            attribute = attribute_dict[attribute_num + 'mainstreaminess']
            edge = str(edge)
            triplet = (user, edge, attribute)
            if triplet in triplets_set:
                continue
            triplets_set.add(triplet)
            file.writelines(user + '\t' + edge + '\t' + attribute + '\n')
            edge_num = edge_num + 1
        print('total collect edge_num ' + str(edge_num))

        for idx, line in enumerate(user_lines):
            if idx == 0:
                continue
            line = line.strip()
            line = line.split('\t')
            if line[11] == '':
                continue
            if len(line[11]) < 3:
                attribute_num = str(0)
            else:
                attribute_num = line[11][2]
            if not user_dict.get(line[0]) or not attribute_dict.get(attribute_num+'cnt_listeningevents'):
                continue
            user = user_dict[line[0]]
            edge = 16
            attribute = attribute_dict[attribute_num + 'cnt_listeningevents']
            edge = str(edge)
            triplet = (user, edge, attribute)
            if triplet in triplets_set:
                continue
            triplets_set.add(triplet)
            file.writelines(user + '\t' + edge + '\t' + attribute + '\n')
            edge_num = edge_num + 1
        print('total collect edge_num ' + str(edge_num))

def create_attribute_yelp(path):
    print('start collect attribute_list')
    with open(path + 'attribute_list.txt', 'w', encoding='utf-8') as file:
        attribute_set = set()
        attribute_num = 0
        file.writelines('attribute_id\tattribute_value\tattribute_type\n')
        print('start collect compliment_funny ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        spilt_range = [0, 1, 2, 3, 4, 5, 8, 10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 100000]
        attribute_list = ['review_count', 'useful', 'funny', 'cool', 'fans', 'compliment_hot', 'compliment_more', 'compliment_profile',
                          'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
                          'compliment_funny', 'compliment_writer', 'compliment_photos']
        for attribute in attribute_list:
            print('start collect' + attribute + str(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            for i in spilt_range:
                num_level = str(i)
                attribute_unin = (num_level, attribute)
                attribute_set.add(attribute_unin)
                file.writelines(num_level + '\t' + str(attribute_num) + '\t' + attribute + '\n')
                attribute_num = attribute_num + 1
            print('total collect attribute_num ' + str(attribute_num))
        attribute = 'average_stars'
        for i in range(0, 51):
            num_level = str("%.1f"%(0.1 * float(i)))
            file.writelines(num_level + '\t' + str(attribute_num) + '\t' + attribute + '\n')
            attribute_num = attribute_num + 1
        attribute_list = ['yelping_since', 'elite']
        for attribute in attribute_list:
            for i in range(2000, 2022):
                num_level = str(i)
                file.writelines(num_level + '\t' + str(attribute_num) + '\t' + attribute + '\n')
                attribute_num = attribute_num + 1

        file.writelines(num_level + '\t' + str(attribute_num) + '\t' + 'friend' + '\n')
        attribute_num = attribute_num + 1
        print('total collect attribute_num ' + str(attribute_num))

def create_attribute_kg_yelp(path):
    print('start collect attribute_kg')
    start_edge = 50

    user_dict = read_user_list(path + 'user_list.txt')
    attribute_dict = read_attribute_list(path + 'attribute_list.txt')
    spilt_range = [0, 1, 2, 3, 4, 5, 8, 10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 100000]
    attribute_list = ['review_count', 'useful', 'funny', 'cool', 'fans', 'compliment_hot', 'compliment_more',
                      'compliment_profile',
                      'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
                      'compliment_funny', 'compliment_writer', 'compliment_photos']
    with open(path + 'attribute_kg.txt', 'w', encoding='utf-8') as file:
        with open(path + 'rawdata/yelp_academic_dataset_user.json', 'r', encoding='utf-8') as f:
            edge_num = 0

            for line in f:
                edge = start_edge
                tmp = json.loads(line)
                # print(tmp)
                user_id = tmp['user_id']
                if user_id not in user_dict:
                    continue
                for i, attribute in enumerate(attribute_list):
                    if attribute not in tmp:
                        continue
                    user_attribute = tmp[attribute]
                    edge = start_edge + i
                    for spilt in spilt_range:
                        if user_attribute <= spilt:
                            file.writelines(user_dict[user_id] + '\t' + str(edge) + '\t' + attribute_dict[str(spilt)+attribute] + '\n')
                            edge_num = edge_num + 1
                            break
                edge = start_edge + len(attribute_list)

                attribute = 'average_stars'
                if attribute in tmp:
                    for i in range(0, 51):
                        spilt = 0.1 * float(i)
                        user_attribute = tmp[attribute]
                        if user_attribute <= spilt:
                            num_level = str("%.1f" % (0.1 * float(i)))
                            file.writelines(user_dict[user_id] + '\t' + str(edge) + '\t' + attribute_dict[
                                str(num_level) + attribute] + '\n')
                            edge_num = edge_num + 1
                            break
                edge = edge + 1

                attribute = 'yelping_since'

                if attribute in tmp:
                    user_attribute = tmp[attribute]
                    if len(user_attribute) != 0:
                        user_attribute = user_attribute.split('-')[0]
                        file.writelines(user_dict[user_id] + '\t' + str(edge) + '\t' + attribute_dict[
                            user_attribute + attribute] + '\n')
                        edge_num = edge_num + 1
                edge = edge + 1

                attribute = 'elite'
                # print(tmp['elite'], type(tmp['elite']))
                if attribute in tmp:
                    user_attribute = tmp[attribute]
                    if len(user_attribute) != 0:
                        user_attribute_list = user_attribute.split(',')
                        for user_attribute in user_attribute_list:
                            file.writelines(user_dict[user_id] + '\t' + str(edge) + '\t' + attribute_dict[
                                user_attribute + attribute] + '\n')
                            edge_num = edge_num + 1
                edge = edge + 1

                print('total collect edge_num ' + str(edge_num))


def change_item_list(path):
    lines = open(path + 'item_list.txt', 'r').readlines()

    with open(path + 'item_list_new.txt', 'w', encoding='utf-8') as file:
        file.writelines('org_id remap_id freebase_id' + '\n')
        for i,line in enumerate(lines):
            if i == 0:
                continue
            l = line.strip()
            tmp = l.split()
            size = len('ydUqgWsF3F27TbauOyib0w')
            file.writelines(tmp[0]+' ' + str(i-1) + ' ' + tmp[1][-size:] + '\n')

def construct_data():

    dataset = 'amazon-book'  # 'amazon-book', 'last-fm', 'yelp2018'

    if dataset == 'amazon-book':
        path = './datasets/amazon-book/'
        # first_reach_amazon(path)
    elif dataset == 'last-fm':
        path = './datasets/last-fm/'
        # create_attribute_lfm(path)
        create_attribute_kg_lfm(path)
        # first_reach_lfm(path)
    elif dataset == 'yelp2018':
        path = './datasets/yelp2018/'
        # change_item_list(path)
        # create_user_item_list(path)
        # create_attribute_yelp(path)
        # create_attribute_kg_yelp(path)
        # first_reach_yelp(path)
    else:
        path = ''
        print('没有找到数据集')

    merge_train_vali_test(path)
    construct_test_scenario(path)
    support_query_set(path)

    # extract_ui_rating_amazon(path)
    # extract_ui_rating_yelp(path)


if __name__ == '__main__':
    construct_data()
