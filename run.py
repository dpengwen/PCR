from lib.config import cfg, args
import numpy as np
import os
import pprint

DEBUG = True
def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch['inp'])
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    if DEBUG:
        print('-------------------------------Evaluating---------------------------------')


    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    
    if DEBUG:
        print('===================================Visualizing=================================')
        import pprint
        print('args:', args)
        print('cfg:')
        pprint.pprint(cfg)

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)


def run_inference():
    from tools import inference
    inference.inference()

def run_inference_fast():
    from tools import inference_fast
    inference_fast.inference()


if __name__ == '__main__':
    print('-----------------------------config---------------------------------')
    print('args:', args)
    print('config:')
    pprint.pprint(cfg)
    print('--------------------------------------------------------------------')
    globals()['run_'+args.type]()
