import argparse
import json
import logging
import os
import random
import gc
import sys
import wandb
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import ImageFolder, CIFAR100, ImageNet, Caltech256
from torchvision.models import get_model, list_models

from datasets.config import Datasets, dataset_config, get_transform
from datasets.custom_dataset import CompositeDataset, ConcatAndSplitDatasets, LoadImageFolders, MCaltech256, \
    collate_wrapper, CustomDataset
from models import get_pretrained_model
from models.anchor_net import AnchorNet
from models.generator import Generator
from models.generator import Encoder, Decoder
from utils.criteria import Compose, CrossEntropy, Accuracy, TopKAccuracy, DistillLoss, \
    Uncertainty, CVAELoss, DecoderLoss, AnchorLoss, KLDivLoss
from utils.logger import logger, print_args, save_results
from utils.trainer import validate, run, Scheduler, Ablation, Baseline, Mode, get_ff, df_train, scheduler_dict, Stage

DEFAULT_SPLIT = [0.8, 0.2]
DA_SPLIT = [0.8, 0.1, 0.1]


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher', type=str, choices=list_models(), default='resnet50')
    parser.add_argument('--teacher_dir', type=str,
                        default='/data/home/xxxx/model/resnet50/office_AW/resnet50_ckpt.pt')
    parser.add_argument('--student', type=str, choices=list_models(), default='resnet18')
    parser.add_argument('--latent_size', type=int, default=256)

    parser.add_argument('--dataset', type=Datasets, choices=list(Datasets), default=Datasets.Office)
    parser.add_argument('--target', type=str, default='/data/home/xxxx/dataset/office/dslr/images')
    parser.add_argument('--test', type=str, default='/data/home/xxxx/dataset/office/dslr/images')

    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--wd', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--a', type=float, default=0.4)
    parser.add_argument('--b', type=float, default=0.2)

    # Generator Arguments
    parser.add_argument('--d_lr', type=float, default=1e-3)
    parser.add_argument('--e_lr', type=float, default=1e-4)
    parser.add_argument('--g_epoch', type=int, default=1000)
    parser.add_argument('--g_dir', type=str, default='', help='Used in case of resume')

    # Anchor Arguments
    parser.add_argument('--a_lr', type=float, default=1e-4)
    parser.add_argument('--a_epoch', type=int, default=500)
    parser.add_argument('--invariant', type=float, default=0.25)
    parser.add_argument('--a_dir', type=str, default='', help='Used in case of resume')

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='/data/home/xxxx/model/GenericKD/')
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--save_checkpoint', default=True, action='store_true')

    parser.add_argument('--sched_type', type=Scheduler, choices=list(Scheduler), default=Scheduler.Linear)
    parser.add_argument('--ablation', type=Ablation, choices=list(Ablation), default=Ablation.NoAblation)
    parser.add_argument('--baseline', type=Baseline, choices=list(Baseline), default=Baseline.NoBaseline)
    parser.add_argument('--s_dir', type=str, help='Used in case of ablation')

    subparsers = parser.add_subparsers(dest='cmd')

    t_parser = subparsers.add_parser('train_teacher')
    t_parser.add_argument('--t_epoch', type=int, default=50)
    t_parser.add_argument('--t_lr', '--t_learning_rate', type=float, default=1e-3)
    t_parser.add_argument('--t_wd', '--t_weight_decay', type=float, default=0)
    t_parser.add_argument('--t_data', type=str, default='/data/home/xxxx/dataset/office/amazon/images', nargs='+')
    t_parser.add_argument('--t_mode', type=Mode, default=Mode.Split, choices=list(Mode))

    args = parser.parse_args()
    date = datetime.now().strftime("%m%d")
    postfix = ''
    if args.ablation != Ablation.NoAblation:
        postfix = '_' + args.ablation.value
    elif args.baseline != Baseline.NoBaseline:
        postfix = '_' + args.baseline.value

    if args.postfix != '' and args.postfix[0] != '_':
        args.postfix = '_' + args.postfix

    if args.cmd is None:
        args.log_dir = os.path.join(args.log_dir, f'{args.teacher}_{args.student}{postfix}_{date}{args.postfix}')

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logger.addHandler(logging.FileHandler(f'{args.log_dir}/result.log', mode='w'))
    # wandb.init(
    #     project="GenericKD",
    #     config=vars(args)
    # )
    print_args(args)
    seed_everything(args.seed)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    setattr(args, 'loader_kwargs', loader_kwargs)
    setattr(args, 'num_classes', dataset_config[args.dataset].num_classes)
    metrics = Compose([Accuracy(), TopKAccuracy(3), TopKAccuracy(5)])

    def best_metric(cur: Dict, prev: Optional[Dict]):
        if (prev is None) or \
            (cur['Acc'] > prev['Acc']) or \
            (cur['Acc'] == prev['Acc'] and cur['Acc@3'] > prev['Acc@3']) or \
            (cur['Acc'] == prev['Acc'] and cur['Acc@3'] == prev['Acc@3'] and cur['Acc@5'] > prev['Acc@5']):
            return True
        else:
            return False

    config = dataset_config[args.dataset]
    if args.cmd == 'train_teacher':
        setattr(args, 'model', args.teacher)
        if args.t_mode == Mode.AutoSplit:
            datasets, t_split = ConcatAndSplitDatasets(LoadImageFolders(args.t_data, args), DEFAULT_SPLIT)
            t_train_data, t_eval_data = datasets
            setattr(args, 'split', t_split)
        elif args.t_mode == Mode.Split:
            assert len(args.t_data) == 2, "Only accept train / val split here"
            t_train_data = ImageFolder(root=args.t_data[0],
                                       transform=get_transform(config.target_size, True, config.normalize))
            t_eval_data = ImageFolder(root=args.t_data[1],
                                      transform=get_transform(config.target_size, False, config.normalize))
        elif args.t_mode == Mode.Torch:
            if args.dataset == Datasets.CIFAR100:
                t_train_data = CIFAR100(root=args.t_data[0], train=True, download=False,
                                        transform=get_transform(config.target_size, True, config.normalize))
                t_eval_data = CIFAR100(root=args.t_data[0], train=False, download=False,
                                       transform=get_transform(config.target_size, False, config.normalize))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        logger.info(f'Start to train teacher {args.teacher} on {args.t_data}')
        logger.info(f'For better performance, use weights pretrained on ImageNet to initialize feature extractors.')

        teacher = get_pretrained_model(args.teacher, config.num_classes)
        t_dataloader = {
            'train': DataLoader(t_train_data, shuffle=True, **loader_kwargs),
            'eval': DataLoader(t_eval_data, shuffle=False, **loader_kwargs)
        }

        optimizer = optim.Adam(teacher.parameters(), lr=args.t_lr)
        run(teacher, criterion=CrossEntropy(), optimizer=optimizer,
            scheduler=ConstantLR(optimizer),
            eval_metrics=metrics, n_epochs=args.t_epoch, do_eval=True,
            save_best=best_metric,
            dataloader=t_dataloader, args=args)
    else:
        args.normalize = True
        if args.resume is None:
            resume: Dict[str, Any] = {'cur_stage': Stage.NoResume}
        else:
            resume = torch.load(args.resume)
        resume['cur_stage'] = Stage(resume['cur_stage'])
        assert resume['cur_stage'] in list(Stage), f'{resume["cur_stage"]} must in {list(Stage)}"'

        setattr(args, 'model', args.student)
        if args.dataset == Datasets.CIFAR100:
            train_data = CIFAR100(root=args.target, train=True, download=False,
                                  transform=get_transform(config.target_size, True, config.normalize))
            eval_data = CIFAR100(root=args.target, train=False, download=False,
                                 transform=get_transform(config.target_size, False, config.normalize))
            test_data = CIFAR100(root=args.test, train=False, download=False,
                                 transform=get_transform(config.target_size, False, config.normalize))
        elif args.target == args.test:
            datasets, split = ConcatAndSplitDatasets(LoadImageFolders([args.target], args, False), DA_SPLIT)
            train_data, eval_data, test_data = datasets
            setattr(args, 'split', split)
        else:
            datasets, split = ConcatAndSplitDatasets(LoadImageFolders([args.target], args, False), DEFAULT_SPLIT)
            train_data, eval_data = datasets
            setattr(args, 'split', split)
            if args.dataset == Datasets.VisDA2017:
                test_data = CustomDataset(root_dir=args.test, transform=get_transform(config.target_size, False, None))
            else:
                test_data = ImageFolder(root=args.test, transform=get_transform(config.target_size, False, None))

        student = get_pretrained_model(args.student, config.num_classes)
        teacher = get_model(args.teacher, num_classes=config.num_classes)
        teacher.load_state_dict(torch.load(args.teacher_dir))

        module_kwargs = {
            'kernel_size': 3,
            'padding': 1,
            'stride': 1
        }
        encoder = Encoder(3, config.num_classes, args.latent_size, [64, 128, 256], config.target_size,
                          scale_strategy=[0.5, 0.5], **module_kwargs)
        decoder = Generator(img_size=config.target_size, latent_size=args.latent_size, num_classes=config.num_classes,
                            in_channels=3, hidden_dims=[256, 128, 64], scale_strategy=[2, 2], **module_kwargs)

        loader = {
            'eval': DataLoader(eval_data, shuffle=False, sampler=None, **loader_kwargs),
            'test': DataLoader(test_data, shuffle=False, sampler=None, **loader_kwargs),
        }
        scheduler = scheduler_dict[args.sched_type](epoch=args.epoch, a=args.a, b=args.b)
        setattr(args, 'scheduler', scheduler)

        if (args.ablation != Ablation.CurriculumKD) or (
            args.ablation == Ablation.NoAblation and args.baseline != Baseline.NoBaseline):
            loader['train'] = DataLoader(train_data, shuffle=True, **loader_kwargs)
        else:
            logger.info(f'Getting uncertainty of train data by teacher')
            logger.info(
                validate(model=teacher, criteria=metrics, loader=DataLoader(train_data, shuffle=False, **loader_kwargs),
                         args=args))
            logger.info(
                validate(model=teacher, criteria=metrics, loader=DataLoader(eval_data, shuffle=False, **loader_kwargs),
                         args=args))
            logger.info(
                validate(model=teacher, criteria=metrics, loader=DataLoader(test_data, shuffle=False, **loader_kwargs),
                         args=args))

            ff = get_ff(teacher, DataLoader(train_data, shuffle=False, **loader_kwargs))
            logger.info(f'ff = {ff}')
            uncertainty = validate(model=teacher,
                                   criteria=Uncertainty(mode=Uncertainty.Energy, num_classes=config.num_classes, ff=ff,
                                                        t=0.9),
                                   # criteria=Uncertainty(mode=Uncertainty.MaxSoftmax),
                                   loader=DataLoader(train_data, shuffle=False, **loader_kwargs),
                                   args=args)
            uncertainty = uncertainty.get_results()
            # Scale uncertainty to [0,1] if needed.
            # Mind that since eval_data is applied with random crop, uncertainty will be slightly different in each run.
            # if uncertainty.min() < 0:
            #     uncertainty -= uncertainty.min().floor()
            # if uncertainty.max() > 1:
            #     uncertainty /= uncertainty.max().ceil()
            uncertainty = uncertainty.detach().cpu()
            # uncertainty = torch.ones_like(uncertainty)

            logger.info(pd.DataFrame(uncertainty).describe())

            train_data = CompositeDataset([train_data, TensorDataset(uncertainty)])
            loader['train'] = DataLoader(train_data, shuffle=True, collate_fn=collate_wrapper, **loader_kwargs)
            setattr(args, 'curriculum', uncertainty.sort())

        s_optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.wd)

        if args.baseline != Baseline.NoBaseline:
            # student.load_state_dict(torch.load(args.s_dir)['student'])
            student.load_state_dict(torch.load(args.s_dir)['state_dict'])

        if args.ablation in [Ablation.NoAblation, Ablation.NoAnchor] and args.baseline == Baseline.NoBaseline:
            # Module 1: Data-Free Learning
            d_optimizer = optim.Adam(decoder.parameters(), args.d_lr)
            e_optimizer = optim.Adam(encoder.parameters(), args.e_lr)

            d_scheduler = CosineAnnealingLR(d_optimizer, T_max=args.g_epoch)
            s_scheduler = CosineAnnealingLR(s_optimizer, T_max=args.g_epoch)
            e_scheduler = CosineAnnealingLR(e_optimizer, T_max=args.g_epoch)

            epoch_offset = 0
            if Stage.Df == resume['cur_stage']:
                epoch_offset = resume['epoch'] + 1
                decoder.load_state_dict(resume['decoder'])
                encoder.load_state_dict(resume['encoder'])
                student.load_state_dict(resume['student'])

                d_scheduler.load_state_dict(resume['sched'][0])
                s_scheduler.load_state_dict(resume['sched'][1])
                e_scheduler.load_state_dict(resume['sched'][2])

                d_optimizer.load_state_dict(resume['optim'][0])
                s_optimizer.load_state_dict(resume['optim'][1])
                e_optimizer.load_state_dict(resume['optim'][2])
                args.best_result = resume['best_result']
                logger.info(f'Resume at Stage 1, Epoch {epoch_offset}/{args.g_epoch}')

            if resume['cur_stage'] in [Stage.NoResume, Stage.Df]:
                decoder, student, encoder = df_train(teacher=teacher,
                                                     student=student,
                                                     encoder=encoder,
                                                     decoder=decoder,
                                                     criterion=(DecoderLoss(t=1), KLDivLoss(), CVAELoss()),
                                                     optimizer=(d_optimizer, s_optimizer, e_optimizer),
                                                     scheduler=(d_scheduler, s_scheduler, e_scheduler),
                                                     n_epochs=args.g_epoch,
                                                     s_val=(loader['test'], metrics, best_metric),
                                                     args=args,
                                                     epoch_offset=epoch_offset)
            else:
                decoder.load_state_dict(torch.load(args.g_dir)['decoder'])
                student.load_state_dict(torch.load(args.g_dir)['student'])
                encoder.load_state_dict(torch.load(args.g_dir)['encoder'])

            # Module 2: Anchor Learning
            anchor = AnchorNet(latent_size=args.latent_size, num_classes=config.num_classes)

            if args.ablation != Ablation.NoAnchor:
                a_optimizer = optim.Adam(anchor.parameters(), lr=args.a_lr)
                a_scheduler = CosineAnnealingLR(a_optimizer, T_max=args.a_epoch)
                setattr(args, 'model', 'anchor')
                setattr(args, 'cur_stage', Stage.Anchor)
                epoch_offset = 0
                if Stage.Anchor == resume['cur_stage']:
                    epoch_offset = resume['epoch'] + 1
                    a_optimizer.load_state_dict(resume['optim'])
                    # a_scheduler.load_state_dict(resume['sched'])
                    anchor.load_state_dict(resume['state_dict'])
                    logger.info(f'Resume at Stage 2, Epoch {epoch_offset}/{args.a_epoch}')

                if resume['cur_stage'] in [Stage.NoResume, Stage.Anchor, Stage.Df]:
                    run(model=anchor,
                        criterion=AnchorLoss(mode='energy', t=1, invariant=args.invariant),
                        optimizer=a_optimizer,
                        scheduler=a_scheduler,
                        n_epochs=args.a_epoch,
                        dataloader=loader,
                        eval_metrics=AnchorLoss(mode='energy', t=1, invariant=args.invariant),
                        do_train=True,
                        do_eval=True,
                        teacher=teacher,
                        decoder=decoder,
                        encoder=encoder,
                        save_best=lambda cur, prev: True if prev is None else (cur['AnchorLoss'] < prev['AnchorLoss']),
                        args=args,
                        epoch_offset=epoch_offset)
                else:
                    anchor.load_state_dict(torch.load(args.a_dir)['state_dict'])
            else:
                logger.info(f'Entering No-Anchor Ablation Mode. Skipping Stage 2...')

            # Module 3: Mixup Learning
            setattr(args, 'model', args.student)
            setattr(args, 'cur_stage', Stage.Distill)
            s_optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.wd)
            s_scheduler = ConstantLR(s_optimizer)
            epoch_offset = 0
            if Stage.Distill == resume['cur_stage']:
                epoch_offset = resume['epoch'] + 1
                s_optimizer.load_state_dict(resume['optim'])
                s_scheduler.load_state_dict(resume['sched'])
                student.load_state_dict(resume['state_dict'])
                logger.info(f'Resume at Stage 3, Epoch {epoch_offset}/{args.epoch}')

            results = run(model=student,
                          criterion=DistillLoss(t=10),
                          optimizer=s_optimizer,
                          scheduler=s_scheduler,
                          eval_metrics=metrics,
                          n_epochs=args.epoch,
                          dataloader=loader,
                          do_eval=True,
                          do_test=True,
                          save_best=best_metric,
                          args=args,
                          teacher=teacher,
                          encoder=encoder,
                          decoder=decoder,
                          anchor=anchor,
                          epoch_offset=epoch_offset)

        elif args.ablation == Ablation.NoKD or args.baseline == Baseline.Finetune:
            logger.info(f'Teacher: {validate(teacher, criteria=metrics, loader=loader["test"], args=args)}')
            results = run(student,
                          criterion=CrossEntropy(),
                          optimizer=s_optimizer,
                          scheduler=ConstantLR(s_optimizer),
                          eval_metrics=metrics,
                          n_epochs=args.epoch,
                          dataloader=loader,
                          do_eval=True,
                          do_test=True,
                          save_best=best_metric, args=args)
        elif args.ablation == Ablation.RawKD or args.baseline == Baseline.KD:
            logger.info(f'Teacher: {validate(teacher, criteria=metrics, loader=loader["test"], args=args)}')
            results = run(student,
                          criterion=DistillLoss(t=10),
                          optimizer=s_optimizer,
                          scheduler=ConstantLR(s_optimizer),
                          eval_metrics=metrics,
                          n_epochs=args.epoch,
                          dataloader=loader,
                          do_eval=True,
                          do_test=True,
                          save_best=best_metric,
                          args=args,
                          teacher=teacher)
        elif args.ablation == Ablation.HyperSearch:
            res = {}
            step_size = 0.1
            for a in np.arange(step_size, 1 + step_size / 10, step_size):
                res[f'{a}'] = {}
                for b in np.arange(0.6, 1 + step_size / 10, step_size):
                    logger.info(f'----------Hyper Setting: a={a},b={b}----------')
                    scheduler = scheduler_dict[args.sched_type](epoch=args.epoch, a=a, b=b)
                    setattr(args, 'scheduler', scheduler)
                    seed_everything(args.seed)
                    anchor = AnchorNet(latent_size=args.latent_size, num_classes=config.num_classes)

                    decoder.load_state_dict(torch.load(args.g_dir)['decoder'])
                    student.load_state_dict(torch.load(args.g_dir)['student'])
                    encoder.load_state_dict(torch.load(args.g_dir)['encoder'])
                    anchor.load_state_dict(torch.load(args.a_dir)['state_dict'])
                    teacher.load_state_dict(torch.load(args.teacher_dir))

                    s_optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.wd)
                    s_scheduler = ConstantLR(s_optimizer)

                    metrics.reset()
                    results = run(model=student,
                                  criterion=DistillLoss(t=10),
                                  optimizer=s_optimizer,
                                  scheduler=s_scheduler,
                                  eval_metrics=metrics,
                                  n_epochs=args.epoch,
                                  dataloader=loader,
                                  do_eval=True,
                                  do_test=True,
                                  save_best=best_metric,
                                  args=args,
                                  teacher=teacher,
                                  encoder=encoder,
                                  decoder=decoder,
                                  anchor=anchor)
                    res[f'{a}'][f'{b}'] = results['test']
            res_json = json.dumps(res)
            with open(os.path.join(args.log_dir, 'hyper_search.json'), 'w') as f:
                f.write(res_json)
        save_results(results['train'], os.path.join(args.log_dir, 'train.csv'))


if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except:
    #     for obj in gc.get_objects():
    #         try:
    #             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #                 logger.info(f'{mode(obj)}, {obj.size()}')
    #         except:
    #             pass
