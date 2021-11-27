from gqa_dataset import *


def calc_statistics():
    with open('{}/full_vocab.json'.format('meta_info/'), 'r') as f:
        vocab = json.load(f)
        ivocab = {v: k for k, v in vocab.items()}

    with open('{}/answer_vocab.json'.format('meta_info/'), 'r') as f:
        answer = json.load(f)
        inv_answer = {v: k for k, v in answer.items()}

    split = 'testdev_pred'
    diagnose = GQA(split=split, mode='val', contained_weight=0.1, threshold=0.0, folder='gqa_bottom_up_features/', cutoff=0.5, vocab=vocab, answer=answer,
                   forbidden='', object_info='meta_info/gqa_objects_merged_info.json', num_tokens=30,
                   num_regions=48, length=9, max_layer=5, distribution=False)

    all_lengths = {}
    for idx, ele in enumerate(diagnose):
        prog = ele[2]
        length = 0
        for one_r in prog:
            if max(one_r) > 0:
                length += 1
        if length not in all_lengths:
            all_lengths[length] = 1
        else:
            all_lengths[length] += 1
        print(f'************************ idx: {idx} ************************')
        print(f"program stats:", all_lengths)


if __name__ == '__main__':
    calc_statistics()
