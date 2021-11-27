from torch import nn
import torch
import torch.optim as optim
import numpy as np
import argparse
import sys
import time
import os
import copy
from torch.autograd import Variable
from torch.utils.data import DataLoader
import glob
import resource
import itertools
from collections import Counter
import pickle
from gqa_dataset import *
from mnnm_models.soft_logic_transformer import *
from workflow.mnnm_arguments import *
from workflow.metrics import *

device = torch.device('cuda')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))


def main():
    args = parse_opt()
    if args.do_submission:
        args.length += 1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.do_finetune:
        lr_decay_step = 2
        args.lr_decrease_start = 2
        args.num_epochs = 12
        lr_decay_epochs = range(args.lr_decrease_start, args.num_epochs, lr_decay_step)
        gradual_warmup_steps = []
    else:
        lr_decay_step = 2
        lr_decay_epochs = range(args.lr_decrease_start, args.num_epochs, lr_decay_step)
        gradual_warmup_steps = [1.0 * args.lr_default, 1.0 *
                                args.lr_default, 1.5 * args.lr_default, 2.0 * args.lr_default]

    print(args)
    repo = os.path.join(args.output, args.id)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if not os.path.exists(repo):
        os.mkdir(repo)

    with open('{}/full_vocab.json'.format(args.meta), 'r') as f:
        vocab = json.load(f)
        ivocab = {v: k for k, v in vocab.items()}

    with open('{}/answer_vocab.json'.format(args.meta), 'r') as f:
        answer = json.load(f)
        inv_answer = {v: k for k, v in answer.items()}

    MAX_LAYER = args.max_layer

    if args.ensemble:
        if args.do_testdev_pred:
            filenames = glob.glob('ensembles/testdev_ensemble_*.pkl')
            print("loading {}".format(filenames))
            ensembles = []
            for filename in filenames:
                with open(filename, 'rb') as f:
                    ensembles.append((filename, pickle.load(f)))

            keys = ensembles[0][1].keys()
            iterator = itertools.combinations(ensembles, 3)
            best_performer = 0
            best_comb = []
            for ens_comb in iterator:
                succ, total = 0, 0
                for k in keys:
                    prob = [ens[1][k]['prob'] for ens in ens_comb]
                    prob = sum(prob) / len(prob)
                    result = prob.argmax()
                    if result == ens_comb[0][1][k]['target']:
                        succ += 1
                    total += 1
                comb = [_[0] for _ in ens_comb]
                acc = succ / total
                #print('the accuracy is {} for {}'.format(acc, comb))
                if acc > best_performer:
                    best_comb = comb
                    best_performer = acc

            print("===========================")
            print(comb, best_performer)
            exit()

        elif args.do_submission:
            filenames = glob.glob('ensembles/mcan/submission_ensemble_*.npy')
            print("loading {}".format(filenames))
            ensembles = []
            for filename in filenames:
                ensembles.append(np.load(filename))

            final_prob = sum(ensembles)
            preds = np.argmax(final_prob, -1)

            with open('ensembles/quesionIds.json', 'r') as f:
                questionIds = json.load(f)

            submissions = []
            for q, pred in zip(questionIds, preds):
                submissions.append({'questionId': q, 'prediction': inv_answer[pred]})

            with open('submission_results.json', 'w') as f:
                json.dump(submissions, f)
            exit()

    print("running model with {}".format(args.model))
    if args.model == "SoftLogicTransformer":
        model = SoftLogicTransformer(vocab_size=len(vocab), stacking=args.stacking, answer_size=len(answer),
                                     visual_dim=args.visual_dim, coordinate_dim=args.additional_dim, n_head=args.n_head,
                                     n_layers=MAX_LAYER, dropout=args.dropout, intermediate_dim=args.num_regions + 1,
                                     pre_layers=args.pre_layers, intermediate_layer=args.intermediate_layer)
        print("Running Modular Transformer model with {} layers with post layer".format(args.stacking))
    else:
        raise NotImplementedError

    if not args.single:
        model = nn.DataParallel(model)
    model.to(device)

    basic_kwargs = dict(length=args.length, object_info=os.path.join(args.meta, args.object_info),
                        num_regions=args.num_regions, distribution=args.distribution,
                        vocab=vocab, answer=answer, max_layer=MAX_LAYER, num_tokens=args.num_tokens,
                        spatial_info='{}/gqa_spatial_merged_info.json'.format(args.meta),
                        forbidden=args.forbidden)

    # generate restriction dict.
    program_type_to_cand = pickle.load(open('meta_info/type2cand_dict.pkl', 'rb'))
    pt_to_cand_mask = {k: torch.zeros(len(answer)) for k in program_type_to_cand}
    for key in program_type_to_cand:
        for ele in program_type_to_cand[key]:
            pt_to_cand_mask[key][ele] = 1

    if args.do_train or args.do_train_all or args.do_finetune:
        if args.do_train_all:
            train_split = 'trainval_all_fully'
            print("using the all programs for bootstrapping")
        else:
            train_split = 'trainval_unbiased_fully'
            print("using the generated programs for training")
        testdev_split = 'testdev_pred'

        if args.do_finetune:
            if args.single:
                model = nn.DataParallel(model)

            model.load_state_dict(torch.load(args.load_from))
            print("Loading the bootstrapped model from {}".format(args.load_from))

        train_dataset = GQA(split=train_split, mode='train', contained_weight=args.contained_weight,
                            threshold=args.threshold, folder=args.data, cutoff=args.cutoff, **basic_kwargs)
        test_dataset = GQA(split=testdev_split, mode='val', contained_weight=args.contained_weight,
                           threshold=args.threshold, folder=args.data, cutoff=args.cutoff, **basic_kwargs)

        if args.num_workers == 1:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)

        cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        KL_loss = KLDivergence()

        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr_default)

        if args.resume != -1:
            filename = glob.glob('{}/{}/model_ep{}*'.format(args.output, args.id, args.resume))[0]
            print("resuming from {}".format(filename))
            model.load_state_dict(torch.load(filename))

        for epoch in range(args.resume + 1, args.num_epochs):
            if epoch < len(gradual_warmup_steps):
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = gradual_warmup_steps[epoch]
                print('lr', optimizer.param_groups[-1]['lr'])
            if epoch in lr_decay_epochs:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] *= args.lr_decay_rate
                print('lr', optimizer.param_groups[-1]['lr'])
            else:
                print('lr', optimizer.param_groups[-1]['lr'])

            model.eval()
            success, total = 0, 0
            for i, batch in enumerate(test_dataloader):
                questionId = batch[-2]
                batch = tuple(Variable(t).to(device) for t in batch[:-2])
                results = model(*batch[:-1])
                if isinstance(results, tuple):
                    logits = results[1]
                else:
                    logits = results
                preds = torch.argmax(logits, -1)
                success_or_not = (preds == batch[-1]).float()

                success += torch.sum(success_or_not).item()
                total += success_or_not.size(0)

            acc = round(success / (total + 0.), 4)
            print("epoch {}, accuracy = {}".format(epoch, acc))

            torch.save(model.state_dict(), os.path.join(repo, 'model_ep{}_{}'.format(epoch, acc)))

            model.train()
            start_time = time.time()
            for i, batch in enumerate(train_dataloader):
                if i == 4999:  # earlier stopping
                    break
                questionId, image_id = batch[-2], batch[-1]
                batch = tuple(Variable(t).to(device) for t in batch[:-2])

                model.zero_grad()
                optimizer.zero_grad()

                results = model(*batch[:-1])
                if isinstance(results, tuple):
                    pre_logits, logits = results
                    length = pre_logits.size(-1)
                    pre_loss = KL_loss(batch[-2].view(-1, length), pre_logits.view(-1, length))
                    pred_loss = cross_entropy(logits, batch[-1])
                else:
                    logits = results
                    pre_loss = torch.FloatTensor([0]).to(device)
                    pred_loss = cross_entropy(logits, batch[-1])

                loss = args.weight * pre_loss + pred_loss

                loss.backward()
                optimizer.step()
                if i % 100 == 0 and i > 0:
                    print("epoch: {}, iteration {}/{}: module loss = {}, pred_loss = {} iter time = {}".
                          format(epoch, i, len(train_dataloader), pre_loss.item(), pred_loss.item(), time.time() - start_time))

                start_time = time.time()

    elif args.do_analyze:
        train_split = 'trainval_unbiased_fully'
        train_dataset = GQA_v2(split=train_split, mode='train', contained_weight=args.contained_weight,
                               threshold=args.threshold, folder=args.data, cutoff=args.cutoff, **basic_kwargs)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)
        KL_loss = KLDivergence()
        if args.load_from != "":
            repo = os.path.join(args.output, args.id, args.load_from)
        print("evaluating model from {}".format(repo))
        model.load_state_dict(torch.load(repo))
        model.eval()
        start_time = time.time()
        for i, batch in enumerate(train_dataloader):
            questionId = batch[-1]
            answerId = batch[-2]
            batch = tuple(Variable(t).to(device) for t in batch[:-1])
            results = model(*batch[:-2])
            pre_logits, logits = results

            length = pre_logits.size(-1)
            pre_loss = KL_loss(batch[-2].view(-1, length), pre_logits.view(-1, length))
            answer_prediction = logits.argmax(-1)

            groundtruth = batch[-2].argmax(-1)
            prediction = pre_logits.topk(1, -1)[1]
            bounding_boxes = batch[7]

            succ, fail = 0, 0
            succ_counter = Counter()
            fail_counter = Counter()
            for progs, idx, gt, pred, ans_pred, ans, bbox, q_id in zip(batch[2], batch[3], groundtruth, prediction, answer_prediction, answerId, bounding_boxes, questionId):
                cur_length = (torch.sum(idx) - 1).long().item()
                for j in range(cur_length):
                    prog = progs[j]
                    if gt[j] in pred[j]:
                        succ += 1
                        succ_counter.update([ivocab[prog[0].item()]])
                    else:
                        fail += 1
                        #print(q_id, bbox[gt[i]], bbox[pred[j][0]])
                        fail_counter.update([ivocab[prog[0].item()]])

                if ans_pred.item() == ans.item():
                    succ_counter.update([ivocab[progs[cur_length][0].item()]])
                else:
                    fail_counter.update([ivocab[progs[cur_length][0].item()]])

            if i == 50:
                break
            else:
                sys.stdout.write("finished {}/40 batches \r".format(i))

        print()
        print(succ_counter, fail_counter)
        print("success rate = {}".format(succ / (succ + fail + 0.)))

    elif args.do_testdev or args.do_testdev_pred or args.do_val:
        split = 'testdev_pred'
        print("Using the split of {}".format(split))
        if args.gqa_loader == 'v2':
            test_dataset = GQA(split=split, mode='val', folder=args.data, cutoff=args.cutoff,
                               threshold=args.threshold, contained_weight=args.contained_weight,
                               pt2cand=pt_to_cand_mask, **basic_kwargs)
        else:
            raise NotImplementedError

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)

        def show(string):
            return [inv_answer[_.item()] for _ in string]

        if args.load_from != "":
            repo = os.path.join(args.output, args.id, args.load_from)
        else:
            filenames = glob.glob(os.path.join(args.output, args.id, 'model_ep*'))
            best = 0
            repo = ""
            for file in filenames:
                acc = float(file.split('_')[-1])
                if acc > best:
                    best = acc
                    repo = file
            args.load_from = repo.split('/')[-1]

        print("evaluating model from {}".format(repo))
        loaded_model = torch.load(repo)
        model.load_state_dict(loaded_model)

        model.eval()
        success, total = 0, 0
        start_idx = 0
        submissions = {}
        ensemble = {}
        failure_id = []
        for i, batch in enumerate(test_dataloader):
            questionId, image_id = batch[-2], batch[-1]
            batch = tuple(Variable(t).to(device) for t in batch[:-2])

            results = model(*batch[:-1])
            if isinstance(results, tuple):
                pre_logits, logits = results
                selected_bbox = torch.argmax(pre_logits[0], -1)
                bounding_bboxes = batch[7][0][:, :4]
            else:
                logits = results

            probs = nn.functional.softmax(logits, -1)
            preds = torch.argmax(probs, -1)

            for qid, prob, t in zip(questionId, probs, batch[-1]):
                ensemble[qid] = {'prob': prob.data.cpu().numpy(), 'target': t.item()}

            if args.debug:
                start = 40
                if i > start:
                    length = len(test_dataset.data[i][3])
                    print(test_dataset.data[i][0], "@", test_dataset.data[i][1], "@", test_dataset.data[i][3], "@",
                          inv_answer[preds[0].item()], "@", inv_answer[test_dataset[i][-2]])

                    print("SELECTION :", selected_bbox)
                    for b in selected_bbox[:length - 1]:
                        if b < args.num_regions:
                            print('[{},{},{},{}]'.format(bounding_bboxes[b][0].item(),
                                                         bounding_bboxes[b][1].item(),
                                                         bounding_bboxes[b][2].item(),
                                                         bounding_bboxes[b][3].item()))
                        elif b == args.num_regions:
                            print(None)
                        elif b == args.num_regions + 1:
                            print("yes")
                        elif b == args.num_regions + 2:
                            print("no")
                        else:
                            print("")
                    print()
                if i == start + 20:
                    break

            for pred, question_id in zip(preds, questionId):
                submissions[question_id] = inv_answer[pred.item()]
            success_or_not = []
            for idx, ele in enumerate(preds):
                cur_pred_inv = inv_answer[ele.item()]
                cur_gt_inv = inv_answer[batch[-1][idx].item()]
                stem_pred, stem_gt = stem_func(cur_pred_inv), stem_func(cur_gt_inv)
                success_or_not.append(stem_pred == stem_gt)
            # success_or_not = (preds == batch[-1]).float()
            success += sum(success_or_not)
            for idx, ele in enumerate(success_or_not):
                if ele < 0.5:   # incorrect
                    failure_id.append([questionId[idx], image_id[idx], inv_answer[preds[idx].item()]])
            failure_p = os.path.join(args.output, args.id, args.id + '_failure.pkl')
            pickle.dump(failure_id, open(failure_p, 'wb'))
            total += len(success_or_not)
            start_idx += args.batch_size

            acc = round(success / (total + 0.), 4)
            print("Validation accuracy =", acc)
        if args.do_testdev:
            with open('/tmp/gt_results.json', 'w') as f:
                json.dump(submissions, f, indent=2)
        else:
            with open('/tmp/pred_results.json', 'w') as f:
                json.dump(submissions, f, indent=2)

    elif args.do_submission:
        test_dataset = GQA(split='submission', mode='val', contained_weight=args.contained_weight,
                           threshold=args.threshold, folder=args.data, cutoff=args.cutoff, **basic_kwargs)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)

        if args.load_from != "":
            repo = os.path.join(args.output, args.id, args.load_from)
        else:
            filenames = glob.glob(os.path.join(args.output, args.id, 'model_ep*'))
            best = 0
            repo = ""
            for file in filenames:
                acc = float(file.split('_')[-1])
                if acc > best:
                    best = acc
                    repo = file
            args.load_from = repo.split('/')[-1]

        print("evaluating model from {}".format(repo))
        model.load_state_dict(torch.load(repo))

        model.eval()
        success, total = 0, 0
        submissions = []
        full_prob = []
        questionIds = []
        for i, batch in enumerate(test_dataloader):
            questionId = batch[-1]
            batch = tuple(Variable(t).to(device) for t in batch[:-1])

            results = model(*batch[:-2])
            if isinstance(results, tuple):
                pre_logits, logits = results
            else:
                logits = results

            if isinstance(results, tuple):
                logits = results[1]
            else:
                logits = results

            prob = torch.softmax(logits, -1)
            preds = torch.argmax(prob, -1)
            questionIds.extend(questionId)
            full_prob.append(prob.data.cpu().numpy())
            for pred, question_id in zip(preds, questionId):
                submissions.append({'questionId': question_id, 'prediction': inv_answer[pred.item()]})
            sys.stdout.write('finished {}/{}\r'.format(i, len(test_dataloader)))
        with open('submission_results.json', 'w') as f:
            json.dump(submissions, f)

    else:
        raise ValueError("Unseen Option")


if __name__ == "__main__":
    main()
