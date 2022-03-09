# encoding: utf-8

from .datasets import build_dataset
from .samplers import build_sampler
from .collate_function import build_collate_fn

from torch.utils.data import DataLoader

def make_data_loader(cfg, is_train):
    if cfg.DATA.DATASETS.NAMES == "none":
        return None, None, None, None

    # 0. config
    dataset_name = cfg.DATA.DATASETS.NAMES
    root_path = cfg.DATA.DATASETS.ROOT_DIR

    # build datasets
    train_set = build_dataset(dataset_name, "train", root_path)
    val_set = build_dataset(dataset_name, "valid", root_path)
    test_set = build_dataset(dataset_name, "test", root_path)

    # build samplers
    sampler_name = cfg.DATA.DATALOADER.SAMPLER
    distributed_world_size = len(cfg.MODEL.DEVICE_ID)

    train_sampler = build_sampler(
        sampler_name, train_set, is_train=is_train,
        distributed_world_size=distributed_world_size
    )
    valid_sampler = build_sampler(
        sampler_name, val_set, is_train=False,
        distributed_world_size=distributed_world_size
    )
    test_sampler = build_sampler(
        sampler_name, test_set, is_train=False,
        distributed_world_size=distributed_world_size
    )

    # build collate function
    collate_fn = build_collate_fn()

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    drop_last = False

    train_batch_size = cfg.TRAIN.DATALOADER.IMS_PER_BATCH
    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )

    val_batch_size = cfg.VAL.DATALOADER.IMS_PER_BATCH
    val_loader = DataLoader(
        val_set, batch_size=val_batch_size, sampler=valid_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )

    test_batch_size = cfg.TEST.DATALOADER.IMS_PER_BATCH
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, sampler=test_sampler,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last
    )

    return train_loader, val_loader, test_loader