from .yacs import CfgNode as CN
import argparse
import os

cfg = CN()

# model
cfg.model = 'hello'
cfg.model_dir = 'data/model'

# network
cfg.network = 'dla_34'

# network heads
cfg.heads = CN()

# task
cfg.task = ''

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = ''
cfg.train.epoch = 140
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

##------- dpwen adding -------##
cfg.backbone_name = 'default'  # 'default' or 'resnet50'                                      
cfg.backbone_out_channel = 256
cfg.network_full_init = False

cfg.use_otpg_flag = True # whether using the oriented text proposal generation module to initialize the contour. defualt is true

cfg.csm_version = 'org'  # 'org', 'csm_v0'. Noting: 'csm_v0' is a flexible version of 'org'. 
cfg.poly_cls_branch = True
cfg.poly_cls_branch_input_type = 'org'  #'feat_diff'
cfg.poly_cls_branch_type = 'org'  # 'org', 'v1'
cfg.poly_conf_thresh = 0.9
cfg.pos_samples_creg = False

cfg.vis_intermediate_output = 'none' # 'none': default, 'htp':vis horizontal text proposal, 'otp', 'clm_1': the first CLM, 'clm_2': the second CLM

cfg.evolve_iter = 2
cfg.snake_conv_type = 'dgrid' #'dgrid': dilated cirConv, 'grid': cirConv, 'sgrid': standard conv


cfg.bpoint_feat_enhance='none'  #'aster' or 'none'  #contour feature enhancement mode
cfg.evolve_ct_feat = False  #whether considering the features of the center point in the evolution 

cfg.ex_type = 'corner'  # corner, extreme
cfg.evolve_init = 'ex'  #'poly' or 'ex'
cfg.hbb_enclose_type = 'poly' # 'obb' means enclose the orented box, otherwise enclose the poly

cfg.snake_net = 'CIA_v4'  #static,dynamic, static_dynamic, static_lstm, CIA_v1...6, 'no_cia'
cfg.psp_size = (1, 16)  #work for CIA_v4
cfg.rs_att_flag = False

cfg.modified_feat_enhance = 'none' # 'ffm' or 'sa'
cfg.final_feat_enhance_type = 'none' #'none', 'lambda', 'sa', 'pc_att', 'double_att', 'ctx_pooling', 'ema' 
cfg.train.mask_center = False
cfg.dist_4d = False
cfg.dist_constraint = False
cfg.centerness = False
cfg.train.ct_reg   = False
cfg.ida_exlayer = 1
##----------------------------##

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1

##----------dpwen-------------##
#cfg.test.target_scale = False
cfg.test.target_scale = (416, 10000)
cfg.rle_nms = False
##----------------------------##

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

cfg.save_ep = 5
cfg.eval_ep = 5

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.05
cfg.demo_path = ''

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    #cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
    #cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    #cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)

    #adding by ffc
    cfg.model = args.model
    cfg.testing_set = args.testing_set
    cfg.record_dir = args.record_dir
    cfg.model_dir   = args.model_dir
    cfg.results_dir = args.results_dir
    cfg.vis_dir = args.vis_dir
    cfg.det_dir = args.det_dir
    cfg.gts_dir = args.gts_dir
    cfg.test.epoch  = args.test_epoch

def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--test_epoch", default=-1, type=int)
parser.add_argument("--model", default="snake", type=str)
parser.add_argument("--model_dir", default="data/model", type=str)
parser.add_argument("--record_dir", default="data/record", type=str)
parser.add_argument("--results_dir", default="results", type=str)
parser.add_argument("--testing_set", default="ctw", type=str)
parser.add_argument("--det_dir", default="det_results", type=str)
parser.add_argument("--vis_dir", default="vis_results", type=str)
parser.add_argument("--gts_dir", default="gts_results", type=str)
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
