import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_preprocess', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_finetune', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_train_all', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_testdev', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_testdev_pred', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_analyze', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_submission', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--ensemble', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--data', type=str, default="gqa_bottom_up_features/",
                        help="whether to train or test the model")
    parser.add_argument('--object_info', type=str, default='gqa_objects_merged_info.json',
                        help="whether to train or test the model")
    parser.add_argument('--object_file', type=str, default="gqa_objects.h5", help="gqa object location")
    parser.add_argument('--spatial_info', type=str, default='gqa_spatial_merged_info.json',
                        help="whether to train or test the model")
    parser.add_argument('--spatial_file', type=str, default="gqa_spatial.h5", help="gqa object location")
    parser.add_argument('--hidden_dim', type=int, default=512, help="whether to train or test the model")
    parser.add_argument('--n_head', type=int, default=8, help="whether to train or test the model")
    parser.add_argument('--pre_layers', type=int, default=3, help="whether to train or test the model")
    parser.add_argument('--glimpse', type=int, default=1, help="whether to train or test the model")
    parser.add_argument('--debug', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--visual_dim', type=int, default=2048, help="whether to train or test the model")
    parser.add_argument('--num_regions', type=int, default=48, help="whether to train or test the model")
    parser.add_argument('--spatial_regions', type=int, default=49, help="whether to train or test the model")
    parser.add_argument('--num_tokens', type=int, default=30, help="whether to train or test the model")
    parser.add_argument('--num_workers', type=int, default=16, help="whether to train or test the model")
    parser.add_argument('--additional_dim', type=int, default=6, help="whether to train or test the model")
    parser.add_argument('--weight', type=float, default=0.5, help="whether to train or test the model")
    parser.add_argument('--batch_size', type=int, default=256, help="whether to train or test the model")
    parser.add_argument('--num_epochs', type=int, default=60, help="num epochs, original is 20, using earlier "
                                                                   "stopping, so it's 60 now.")
    parser.add_argument('--max_layer', type=int, default=5, help="whether to train or test the model")
    parser.add_argument('--single', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--intermediate_layer', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--model', type=str, default="Tree", help="whether to train or test the model")
    parser.add_argument('--stacking', type=int, default=6, help="whether to train or test the model")
    parser.add_argument('--threshold', type=float, default=0., help="whether to train or test the model")
    parser.add_argument('--cutoff', type=float, default=0.5, help="whether to train or test the model")
    parser.add_argument('--resume', type=int, default=-1, help="whether to train or test the model")
    parser.add_argument('--concept_glove', type=str, default="models/concept_emb.npy",
                        help="whether to train or test the model")
    parser.add_argument('--word_glove', type=str, default="meta_info/en_emb.npy",
                        help="whether to train or test the model")
    parser.add_argument('--dropout', type=float, default=0.1, help="whether to train or test the model")
    parser.add_argument('--distribution', default=False, action='store_true', help="whether to train or test the model")
    parser.add_argument('--load_from', type=str, default="", help="whether to train or test the model")
    parser.add_argument('--output', type=str, default="models", help="whether to train or test the model")
    parser.add_argument('--length', type=int, default=9, help="whether to train or test the model")
    parser.add_argument('--id', type=str, default="default", help="whether to train or test the model")
    parser.add_argument('--groundtruth', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--gqa_loader', type=str, default="v2", help="whether to train or test the model")
    parser.add_argument('--lr_decrease_start', type=int, default=10, help="whether to train or test the model")
    parser.add_argument('--ensemble_id', type=int, default=1, help="whether to train or test the model")
    parser.add_argument('--lr_default', type=float, default=1e-4, help="whether to train or test the model")
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help="whether to train or test the model")
    parser.add_argument('--contained_weight', type=float, default=0.1, help="whether to train or test the model")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--meta', default="meta_info/", type=str, help="The hidden size of the state")
    parser.add_argument('--forbidden', default="", type=str, help="The hidden size of the state")

    args = parser.parse_args()
    return args
