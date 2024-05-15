import pandas as pd
import gzip
import numpy as np
import json
import tqdm
import random
import collections
import time

random.seed(2020)

def read_user_list(path):
    """
    return: dict{org_id: remap_id} type: {str: str}
    """
    lines = open(path, 'r').readlines()
    user_dict = dict()
    max_user = 0
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        l = line.strip()
        tmp = l.split()
        user_dict[tmp[0]] = tmp[1]
        max_user = max(max_user, int(tmp[1]))

    return user_dict, max_user


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


def create_attribute_kg_lfm(path):
    print('start collect attribute_kg')
    user_dict ,max_user= read_user_list(path + 'user_list.txt')
    print(len(user_dict))
    max_user = max_user + 1
    attribute_dict = read_attribute_list(path + 'attribute_list.txt')
    with open(path + 'attribute_list_final.txt', 'w') as f:
        f.writelines('org_id remap_id\n')
        for user in user_dict:
            f.writelines(user + ' ' + user_dict[user] + '\n')
        for attribute in attribute_dict:
            f.writelines(attribute + ' ' + str(int(attribute_dict[attribute])+ max_user) + '\n')
    triples_lines = open(path+'attribute_kg.txt', 'r').readlines()
    with open(path + 'attribute_kg_final.txt', 'w') as f:
        for idx, line in enumerate(triples_lines):
            l = line.strip()
            temp = l.split()
            temp[2] = str(int(temp[2]) + max_user)
            f.writelines(temp[0] + ' ' + temp[1] + ' ' + temp[2] + '\n')


def construct_user_data():

    dataset = 'yelp2018'  # 'amazon-book', 'last-fm', 'yelp2018'
    if dataset == 'amazon-book':
        path = './datasets/amazon-book/'
    elif dataset == 'last-fm':
        path = './datasets/last-fm/'
        create_attribute_kg_lfm(path)
    elif dataset == 'yelp2018':
        path = './datasets/yelp2018/'
        create_attribute_kg_lfm(path)
    else:
        path = ''
        print('没有找到数据集')
    print(path)


if __name__ == '__main__':
    construct_user_data()