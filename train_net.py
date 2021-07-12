from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch.multiprocessing

def setup_seed(seed=3):
    import random, os
    import numpy as np 
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEBUG = True
def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    #evaluator = make_evaluator(cfg)

    if cfg.network_full_init:
        begin_epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
        begin_epoch = 0
    else:
        begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)

    # set_lr_scheduler(cfg, scheduler)
    if DEBUG:
        print('------------------Loading training set-------------------')
    train_loader = make_data_loader(cfg, is_train=True)
    if DEBUG:
        print('Loading training set done...')
        print('---------------------------------------------------------')
    #val_loader = make_data_loader(cfg, is_train=False)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        #if (epoch + 1) % cfg.eval_ep == 0:
        #    trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def main():
    if DEBUG:
        print('------------------------------------------------------------')
        print('args',args)
        print('cfg:')
        import pprint
        pprint.pprint(cfg)
        print('------------------------------------------------------------')
    setup_seed()

    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    main()
