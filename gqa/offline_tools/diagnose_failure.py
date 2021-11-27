from gqa_dataset import *


def diagnose_failure():
    with open('{}/full_vocab.json'.format('meta_info/'), 'r') as f:
        vocab = json.load(f)
        ivocab = {v: k for k, v in vocab.items()}

    with open('{}/answer_vocab.json'.format('meta_info/'), 'r') as f:
        answer = json.load(f)
        inv_answer = {v: k for k, v in answer.items()}

    split = 'testdev_pred'
    failure_p = os.path.join("/lhddscratch/zelin/soft_vqa/gqa/cycle_mnnm/models/TreeSparsePost2Full", "TreeSparsePost2Full_failure.pkl")
    diagnose = GQA(split=split, mode='val', contained_weight=0.1, threshold=0.0, folder='gqa_bottom_up_features/', cutoff=0.5, vocab=vocab, answer=answer,
                   forbidden='', object_info='meta_info/gqa_objects_merged_info.json', num_tokens=30,
                   num_regions=48, length=9, max_layer=5, distribution=False, failure_path=failure_p)

    for idx, ele in enumerate(diagnose.data):
        question_id, image_id, fake_answer = ele[0], ele[1], ele[2]
        cur_p = os.path.join('mmnm_questions/', 'mmnm_{}.pkl'.format(image_id))
        entry = pickle.load(open(cur_p, 'rb'))[question_id]
        print(f'************************ idx: {idx}, qid: {question_id}, image id: {image_id} ************************')
        print(f"question: {entry[1]}.")
        print(f"program:")
        print(entry[3])
        print(f"gt answer: {entry[-1]}.")
        print(f"pred answer: {fake_answer}.")


if __name__ == '__main__':
    diagnose_failure()
