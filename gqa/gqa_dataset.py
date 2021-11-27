from nltk.tokenize import word_tokenize
import json
import h5py
from gqa_mnnm_constants import *
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
import time
from torch import nn


class GQA(Dataset):
    def __init__(self, **args):
        self.mode = args['mode']
        self.split = args['split']
        self.vocab = args['vocab']
        self.answer_vocab = args['answer']
        self.num_tokens = args['num_tokens']
        self.num_regions = args['num_regions']
        self.LENGTH = args['length']
        self.MAX_LAYER = args['max_layer']
        self.folder = args['folder']
        self.threshold = args['threshold']
        self.contained_weight = args['contained_weight']
        self.cutoff = args['cutoff']
        self.distribution = args['distribution']
        self.failure_p = args['failure_path'] if 'failure_path' in args else None

        if args['forbidden'] != '':
            with open(args['forbidden'], 'r') as f:
                self.forbidden = json.load(f)
            self.forbidden = set(self.forbidden)
        else:
            self.forbidden = set([])
        if self.failure_p is not None:
            print(f"Loading failed data from {self.failure_p}.")
            self.data = pickle.load(open(self.failure_p, 'rb'))
        else:
            meta_list_p = os.path.join('mmnm_questions/', 'list_' + self.split + ".pkl")
            print(f"Loading meta data from {meta_list_p}.")
            self.data = pickle.load(open(meta_list_p, 'rb'))

        with open(args['object_info']) as f:
            self.object_info = json.load(f)
        print(f"there are in total {len(self.data)} instances.")

    def __getitem__(self, index):
        if self.failure_p:
            question_id, image_id = self.data[index]
        else:
            image_id, question_id = self.data[index]
        cur_p = os.path.join('mmnm_questions/', 'mmnm_{}.pkl'.format(image_id))
        entry = pickle.load(open(cur_p, 'rb'))[question_id]
        obj_info = self.object_info[entry[0]]
        if not entry[0].startswith('n'):
            if len(entry[0]) < 7:
                entry[0] = "0" * (7 - len(entry[0])) + entry[0]

        image_id = entry[0]
        question = entry[1]
        inputs = entry[3]
        prog_key = inputs[-1][0]
        connection = entry[4]
        length = min(len(inputs), self.LENGTH)

        # Prepare Question
        idxs = word_tokenize(question)[:self.num_tokens]
        question = [self.vocab.get(_, UNK) for _ in idxs]
        question += [PAD] * (self.num_tokens - len(idxs))
        question = np.array(question, 'int64')

        question_masks = np.zeros((len(question),), 'float32')
        question_masks[:len(idxs)] = 1.

        # Prepare Program
        program = np.zeros((self.LENGTH, 8), 'int64')
        depth = np.zeros((self.LENGTH,), 'int64')
        for i in range(length):
            for j, text in enumerate(inputs[i]):
                if text is not None:
                    program[i][j] = self.vocab.get(text, UNK)

        # Prepare Program mask
        program_masks = np.zeros((self.LENGTH,), 'float32')
        program_masks[:length] = 1.

        # Prepare Program Transition Mask
        transition_masks = np.zeros((self.MAX_LAYER, self.LENGTH, self.LENGTH), 'uint8')
        activate_mask = np.zeros((self.MAX_LAYER, self.LENGTH), 'float32')
        for i in range(self.MAX_LAYER):
            if i < len(connection):
                for idx, idy in connection[i]:
                    transition_masks[i][idx][idy] = 1
                    depth[idx] = i
                    activate_mask[i][idx] = 1
            for j in range(self.LENGTH):
                if activate_mask[i][j] == 0:
                    # As a placeholder
                    transition_masks[i][j][j] = 1
                else:
                    pass
        vis_mask = np.zeros((self.num_regions,), 'float32')
        # new_bottom_up = self.all_bottom_up[image_id]

        # Prepare Vision Feature
        bottom_up = np.load(os.path.join(self.folder, 'gqa_{}.npz'.format(image_id)))
        adaptive_num_regions = min((bottom_up['conf'] > self.threshold).sum(), self.num_regions)

        # Cut off the bottom up features
        object_feat = bottom_up['features'][:adaptive_num_regions]
        bbox_feat = bottom_up['norm_bb'][:adaptive_num_regions]
        vis_mask[:bbox_feat.shape[0]] = 1.

        # Padding zero
        if object_feat.shape[0] < self.num_regions:
            padding = self.num_regions - object_feat.shape[0]
            object_feat = np.concatenate([object_feat, np.zeros(
                (padding, object_feat.shape[1]), 'float32')], 0)
        if bbox_feat.shape[0] < self.num_regions:
            padding = self.num_regions - bbox_feat.shape[0]
            bbox_feat = np.concatenate([bbox_feat, np.zeros(
                (padding, bbox_feat.shape[1]), 'float32')], 0)
        num_regions = bbox_feat.shape[0]

        # exist = np.full((self.LENGTH, ), -1, 'float32')
        returns = entry[2]
        intermediate_idx = np.full(
            (self.LENGTH, num_regions + 1), 0, 'float32')
        intersect_iou = np.full(
            (length - 1, num_regions + 1), 0., 'float32')
        if self.mode == 'train':
            for idx in range(length - 1):
                if isinstance(returns[idx], list):
                    if returns[idx] == [-1, -1, -1, -1]:
                        intermediate_idx[idx][num_regions] = 1
                    else:
                        gt_coordinate = (returns[idx][0] / (obj_info['width'] + 0.),
                                         returns[idx][1] / (obj_info['height'] + 0.),
                                         (returns[idx][2] + returns[idx][0]) / (obj_info['width'] + 0.),
                                         (returns[idx][3] + returns[idx][1]) / (obj_info['height'] + 0.))
                        for i in range(num_regions):
                            intersected, contain = intersect(gt_coordinate, bbox_feat[i, :4], True, 'x1y1x2y2')
                            intersect_iou[idx][i] = intersected  # + self.contained_weight * contain

                        # if self.distribution:
                        # mask = (intersect_iou[idx] > self.cutoff).astype('float32')
                        # intersect_iou[idx] *= mask
                        intermediate_idx[idx] = intersect_iou[idx] / (intersect_iou[idx].sum() + 0.001)
                        # else:
                        #    intermediate_idx[idx] = (intersect_iou[idx] > self.cutoff).astype('float32')
                        #    intermediate_idx[idx] = intermediate_idx[idx] / (intermediate_idx[idx].sum() + 0.001)
        else:
            intermediate_idx = 0
        # Prepare index selection
        index = length - 1
        # Prepare answer
        answer_id = self.answer_vocab.get(entry[-1], UNK)

        return question, question_masks, program, program_masks, transition_masks, activate_mask, object_feat, \
               bbox_feat, vis_mask, index, depth, intermediate_idx, answer_id, question_id, image_id

    def __len__(self):
        return len(self.data)


def create_splited_questions(dataset, save_dir='mmnm_questions/'):
    for idx, entry in enumerate(dataset.data):
        print(f"[{dataset.mode}]processing idx {idx} ...", end='\r')
        image_id = entry[0]
        questionId = entry[-2]
        save_p = os.path.join(save_dir, 'mmnm_{}.pkl'.format(image_id))
        if os.path.exists(save_p):
            cur_meta = pickle.load(open(save_p, 'rb'))
        else:
            cur_meta = {}
        cur_meta[questionId] = entry
        pickle.dump(cur_meta, open(save_p, 'wb'))


def generate_meta_list(dataset):
    data_list = []
    for idx, entry in enumerate(dataset.data):
        print(f"[{dataset.split}]processing idx {idx} ...", end='\r')
        image_id = entry[0]
        questionId = entry[-2]
        data_list.append((image_id, questionId))
    save_p = os.path.join('mmnm_questions/', 'list_' + dataset.split + ".pkl")
    pickle.dump(data_list, open(save_p, 'wb'))


def test_dataset():
    with open('{}/full_vocab.json'.format('meta_info/'), 'r') as f:
        vocab = json.load(f)
        ivocab = {v: k for k, v in vocab.items()}

    with open('{}/answer_vocab.json'.format('meta_info/'), 'r') as f:
        answer = json.load(f)
        inv_answer = {v: k for k, v in answer.items()}

    train_dataset = GQA(split='trainval_all_fully', mode='train', contained_weight=0.1,
                        threshold=0.0, folder='gqa_bottom_up_features/', cutoff=0.5, vocab=vocab, answer=answer,
                        forbidden='', object_info='meta_info/gqa_objects_merged_info.json', num_tokens=30,
                        num_regions=48, length=9, max_layer=5, distribution=False)
    test_d = train_dataset[0]
    print(test_d)


