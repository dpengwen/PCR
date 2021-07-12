from lib.config import cfg
class DatasetCatalog(object):
    dataset_attrs = {
        'CtwTrain': {
            'id': 'ctw',
            'data_root': 'data/ctw/train/text_image',
            'ann_file': 'data/ctw/train/ctw_train_instance.json',
            'split': 'train'
        },
        'CtwMini': {
            'id': 'ctw',
            'data_root': 'data/ctw/train/text_image',
            'ann_file': 'data/ctw/train/ctw_train_instance.json',
            'split': 'mini'
        },
        'TotTrain': {
            'id': 'tot',
            'data_root': 'data/tot/train/images',
            'ann_file': 'data/tot/train/tot_train_instance.json',
            'split': 'train'
        },
        'TotMini': {
            'id': 'tot',
            'data_root': 'data/tot/train/images',
            'ann_file': 'data/tot/train/tot_train_instance.json',
            'split': 'mini'
        },
        'ArtTrain': {
            'id': 'art',
            'data_root': 'data/art/train/images',
            'ann_file': 'data/art/train/art_train_instance.json',
            'split': 'train'
        },
        'ArtMini': {
            'id': 'art',
            'data_root': 'data/art/train/images',
            'ann_file': 'data/art/train/art_train_instance.json',
            'split': 'mini'
        },
        'MsraTrain': {
            'id': 'msra',
            'data_root': 'data/msra/train/images',
            'ann_file': 'data/msra/train/msra_train_instance.json',
            'split': 'train'
        },
        'MsraMini': {
            'id': 'msra',
            'data_root': 'data/msra/train/images',
            'ann_file':  'data/msra/train/msra_train_instance.json',
            'split': 'mini'
        },
        'IC15Train': {
            'id': 'ic15',
            'data_root': 'data/ic15/train/images',
            'ann_file': 'data/ic15/train/ic15_train_instance.json',
            'split': 'train'
        },
        'IC15Mini': {
            'id': 'ic15',
            'data_root': 'data/ic15/train/images',
            'ann_file':  'data/ic15/train/ic15_train_instance.json',
            'split': 'mini'
        },
        'MltTrain': {
            'id': 'mlt',
            'data_root': 'data/mlt17/train/images',
            'ann_file': 'data/mlt17/train/mlt_train_instance.json',
            'split': 'train'
        },
        'MltMini': {
            'id': 'mlt',
            'data_root': 'data/mlt17/train/images',
            'ann_file': 'data/mlt17/train/mlt_train_instance.json',
            'split': 'mini'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()

