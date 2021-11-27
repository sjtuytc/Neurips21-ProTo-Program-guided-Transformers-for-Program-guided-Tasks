# generate candidate answers given programs
import time
from gqa_dataset import *


def generate_dicts(encode=True):
    with open('{}/full_vocab.json'.format('meta_info/'), 'r') as f:
        vocab = json.load(f)

    with open('{}/answer_vocab.json'.format('meta_info/'), 'r') as f:
        answer = json.load(f)

    split = 'trainval_all_fully'
    mode = 'train'
    gqa_d = GQA(split=split, mode=mode, contained_weight=0.1, threshold=0.0, folder='gqa_bottom_up_features/',
                cutoff=0.5, vocab=vocab, answer=answer, forbidden='', object_info='meta_info/gqa_objects_merged_info.json',
                num_tokens=30, num_regions=48, length=9, max_layer=5, distribution=False, failure_path=None)

    type2cand_dict = {}
    start_t = time.time()
    for idx, ele in enumerate(gqa_d.data):
        if idx % 1000 == 0:
            time_per_iter = (time.time() - start_t) / (idx + 1e-9)
            print(f"{idx} / {len(gqa_d.data)}, finished. Time per iter: {time_per_iter:.3f}.", end='\r')
            type2cand_dict_p = os.path.join('meta_info/type2cand_dict.pkl')
            pickle.dump(type2cand_dict, open(type2cand_dict_p, 'wb'))
        image_id, question_id = ele[0], ele[1]
        cur_p = os.path.join('mmnm_questions/', 'mmnm_{}.pkl'.format(image_id))
        entry = pickle.load(open(cur_p, 'rb'))[question_id]
        # prog_type = [ele for ele in entry[3][-1] if ele is not None]
        # prog_type = '_'.join(prog_type)
        prog_type = entry[3][-1][0]
        answer = entry[-1]
        if encode:
            answer = gqa_d.answer_vocab.get(answer, UNK)
        if prog_type not in type2cand_dict:
            type2cand_dict[prog_type] = set()
        if answer not in type2cand_dict[prog_type]:
            type2cand_dict[prog_type].add(answer)


if __name__ == '__main__':
    generate_dicts()
