import os
from argparse import Namespace
from copy import deepcopy
from enum import Enum
from typing import Optional, Dict, Union, Tuple, Any, Callable, List

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.transforms.functional import normalize
from tqdm import tqdm

from datasets.config import dataset_config
from utils.criteria import Metric, Compose
from utils.hook import FeatureMeanVarHook
from utils.logger import logger
from utils.scheduler import LinearScheduler, StepScheduler


class Ablation(Enum):
    NoAblation = 'no_ablation'
    NoKD = 'no_kd'
    RawKD = 'raw_kd'
    CurriculumKD = 'curriculum_kd'
    HyperSearch = 'hyper_search'
    NoAnchor = 'no_anchor'


class Baseline(Enum):
    NoBaseline = 'no_baseline'
    Finetune = 'finetune'
    KD = 'kd'


class Scheduler(Enum):
    Linear = 'LinearScheduler'
    Step = 'StepScheduler'


class Mode(Enum):
    Torch = 'torch'
    AutoSplit = 'auto_split'
    Split = 'split'


class Stage(Enum):
    Df = 'df'
    Anchor = 'anchor'
    Distill = 'distill'
    NoResume = 'no_resume'


scheduler_dict = {
    Scheduler.Linear: LinearScheduler,
    Scheduler.Step: StepScheduler,
}


def set_eval(models: List[Optional[nn.Module]], device: str):
    for model in models:
        if model is not None:
            model.to(device)
            model.eval()


def set_train(models: List[Optional[nn.Module]], device: str):
    for model in models:
        if model is not None:
            model.to(device)
            model.train()


def to(optimizers: List[Optimizer], device: str):
    for o in optimizers:
        for state in o.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(device)


def train(epoch: int, model: nn.Module, criterion: Metric, optimizer: Optimizer,
          scheduler: Any,
          loader: DataLoader,
          eval_metrics: Optional[Metric] = None,
          teacher: Optional[nn.Module] = None,
          encoder: Optional[nn.Module] = None,
          decoder: Optional[nn.Module] = None,
          anchor: Optional[nn.Module] = None,
          args: Optional[Namespace] = None, device: str = 'cuda:0', show_progress: bool = True) -> Dict:
    config = dataset_config[args.dataset]

    set_train([model], device)
    set_eval([teacher, encoder, decoder, anchor], device)

    if show_progress:
        loader = tqdm(loader, total=len(loader))

    criterion.reset()
    if eval_metrics is not None:
        eval_metrics.reset()

    tot_loss = 0

    with torch.set_grad_enabled(True):
        for batch_idx, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            y = batch[1]

            if teacher is not None:
                if anchor is not None:
                    s = getattr(args, 's')
                    z = encoder(x, labels=y)
                    if args.ablation == Ablation.NoAnchor:
                        zz = z
                    else:
                        zz, _ = anchor(z, labels=y)
                    z = (1 - s) * zz + s * z
                    img = decoder(z, labels=y)
                    img = (1 - s) * img + s * x
                    n_img = normalize(img, *config.normalize)

                    x = torch.cat([x, n_img])
                    y = torch.cat([y, y])
                    outputs = model(x)
                    t_outputs = model(x)

                    loss = criterion.update(outputs, y, t_outputs=t_outputs).get_results()
                    if eval_metrics is not None:
                        with torch.no_grad():
                            eval_metrics.update(outputs, y, t_outputs=t_outputs)
                elif encoder is not None and decoder is not None:
                    z = encoder(x, labels=y)
                    z, mask = model(z, labels=y)
                    img = decoder(z, labels=y)
                    n_img = normalize(img, *config.normalize)
                    outputs = teacher(n_img)

                    loss = criterion.update(outputs, y, mask=mask).get_results()
                    if eval_metrics is not None:
                        with torch.no_grad():
                            eval_metrics.update(outputs, y, mask=mask)
                else:
                    if hasattr(args, 'normalize') and args.normalize:
                        x = normalize(x, *config.normalize)
                    outputs = model(x)
                    t_outputs = teacher(x)

                    loss = criterion.update(outputs, y, t_outputs=t_outputs).get_results()
                    if eval_metrics is not None:
                        with torch.no_grad():
                            eval_metrics.update(outputs, y, t_outputs=t_outputs)
            else:
                if hasattr(args, 'normalize') and args.normalize:
                    x = normalize(x, *config.normalize)
                outputs = model(x)
                loss = criterion.update(outputs, y).get_results()
                if eval_metrics is not None:
                    with torch.no_grad():
                        eval_metrics.update(outputs, y)

            tot_loss += loss.detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            criterion.reset()

            if show_progress:
                loader.set_description('epoch: {:04} loss = {:8.3f} {}'.format(epoch, tot_loss, eval_metrics))
    return {
        'epoch': epoch,
        'loss': tot_loss,
        **eval_metrics.get_results()
    } if eval_metrics is not None else {
        'epoch': epoch,
        'loss': tot_loss
    }


def validate(model: nn.Module, criteria: Metric, loader: DataLoader,
             teacher: Optional[nn.Module] = None,
             encoder: Optional[nn.Module] = None,
             decoder: Optional[nn.Module] = None,
             args: Optional[Namespace] = None, device: str = 'cuda:0', show_progress: bool = True,
             return_results: bool = False) -> Union[Metric, Tuple[Metric, Tensor]]:
    set_eval([model, teacher, encoder, decoder], device)

    if show_progress:
        loader = tqdm(loader, total=len(loader))
    config = dataset_config[args.dataset]

    criteria.reset()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            if hasattr(args, 'normalize') and args.normalize:
                x = normalize(x, *config.normalize)
            y = batch[1]

            if args.model == 'anchor':
                z = encoder(x, labels=y)
                z, mask = model(z, labels=y)
                img = decoder(z, labels=y)
                n_img = normalize(img, *config.normalize)
                outputs = teacher(n_img)

                criteria.update(outputs, y, mask=mask)
                results.append(outputs)
            else:
                outputs = model(x)
                criteria.update(outputs, y)
                results.append(outputs)

        results = torch.cat(results).to(device).softmax(dim=-1)
        return (criteria, results) if return_results else criteria


def run(model: nn.Module, criterion: Metric, optimizer: Optimizer,
        scheduler: Any,
        n_epochs: int, dataloader: Dict[str, DataLoader], eval_metrics: Optional[Metric] = None,
        do_train: bool = True, do_eval: bool = False, do_test: bool = False,
        teacher: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        anchor: Optional[nn.Module] = None,
        save_best: Callable[[Any, Any], bool] = None,
        args: Optional[Namespace] = None, device: str = 'cuda:0', epoch_offset: int = 0) -> Dict:
    assert (('train' in dataloader.keys()) or not do_train) and \
           (('eval' in dataloader.keys()) or not do_eval) and \
           (('test' in dataloader.keys()) or not do_test) and \
           (do_train or do_eval or do_test)
    cur_stage = args.cur_stage.value if hasattr(args, 'cur_stage') else model.__class__.__name__.lower()
    ckpt_dir = f'{cur_stage}_stat_ckpt.pt'
    if do_eval and not isinstance(eval_metrics, Compose):
        eval_metrics = Compose([eval_metrics])

    train_results = []
    # to([optimizer], device)
    train_loader = dataloader['train']
    train_data = train_loader.dataset
    best_result = None
    res = {}
    for epoch in range(epoch_offset, n_epochs):
        if do_train:
            if args.ablation == Ablation.CurriculumKD:
                s = args.scheduler.threshold(epoch)
                idx = args.curriculum.indices[:int(s * len(args.curriculum.indices))].sort().values
                train_loader = DataLoader(Subset(train_data, idx), **args.loader_kwargs)
                setattr(args, 's', s)
            elif (args.ablation in [Ablation.NoAblation, Ablation.HyperSearch, Ablation.NoAnchor]) and \
                args.baseline == Baseline.NoBaseline and args.cmd is None:
                s = args.scheduler.threshold(epoch)
                setattr(args, 's', s)
            results = train(epoch=epoch, model=model, criterion=criterion, optimizer=optimizer, loader=train_loader,
                            eval_metrics=eval_metrics, args=args, device=device, scheduler=scheduler,
                            teacher=teacher,
                            encoder=encoder,
                            decoder=decoder,
                            anchor=anchor)
            train_results.append(results)
        if do_eval:
            eval_metrics = validate(model=model, criteria=eval_metrics, loader=dataloader['eval'], args=args,
                                    device=device, show_progress=False,
                                    teacher=teacher,
                                    encoder=encoder,
                                    decoder=decoder,
                                    return_results=False)
            logger.info(f'Eval [{epoch}/{n_epochs}]:{eval_metrics}')

        if args.save_checkpoint:
            result = eval_metrics.get_results() if do_eval else train_results[-1]['loss']
            if save_best is None or save_best(result, best_result):
                if save_best:
                    logger.info(f'Best is saved at epoch {epoch}')
                best_result = result
                res = {
                    'epoch': epoch,
                    'optim': optimizer.state_dict(),
                    'sched': scheduler.state_dict(),
                    'metrics': eval_metrics.get_results() if eval_metrics is not None else '',
                    'cur_stage': cur_stage,
                    'state_dict': model.state_dict(),
                }
                if hasattr(args, 'split'):
                    res['split'] = args.split
                torch.save(res, os.path.join(args.log_dir, ckpt_dir))
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'{args.model}_ckpt.pt'))

    res['train'] = train_results
    if args.save_checkpoint and epoch_offset < n_epochs:
        model.load_state_dict(torch.load(os.path.join(args.log_dir, f'{args.model}_ckpt.pt')))
    if do_test:
        eval_metrics = validate(model=model, criteria=eval_metrics, loader=dataloader['eval'], args=args,
                                device=device, show_progress=False,
                                teacher=teacher,
                                encoder=encoder,
                                decoder=decoder,
                                return_results=False)
        logger.info(f'Eval [best]:{eval_metrics}')
        eval_metrics, results = validate(model=model, criteria=eval_metrics, loader=dataloader['test'],
                                         args=args,
                                         teacher=teacher,
                                         encoder=encoder,
                                         decoder=decoder,
                                         device=device, show_progress=False,
                                         return_results=True)
        logger.info(f'Test:{eval_metrics}')
        res['test'] = eval_metrics.get_results()
        results = results.clone().detach().cpu().numpy()

        with open(os.path.join(args.log_dir, 'results.txt'), 'w') as f:
            for i in range(results.shape[0]):
                print(results[i], file=f)

    try:
        ckpt = torch.load(os.path.join(args.log_dir, ckpt_dir))
        ckpt['epoch'] = n_epochs
        torch.save(ckpt, os.path.join(args.log_dir, ckpt_dir))
    except FileNotFoundError:
        pass
    return res


def get_ff(model: nn.Module, loader: DataLoader, device: str = 'cuda:0') -> Tuple[Tensor, Tensor]:
    model.to(device)
    model.eval()

    loader = tqdm(loader, total=len(loader))
    fmin = 1e5
    fmax = -1e5
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            outputs = model(x)
            max_logits = outputs.max(dim=1).values
            if fmin > max_logits.min():
                fmin = max_logits.min()
            if fmax < max_logits.max():
                fmax = max_logits.max()
    return fmin.floor(), fmax.ceil()


def df_train(teacher: nn.Module, student: nn.Module, encoder: nn.Module, decoder: nn.Module,
             criterion: Tuple[Metric, Metric, Metric],
             optimizer: Tuple[Optimizer, Optimizer, Optimizer],
             scheduler: Tuple[Any, Any, Any],
             n_epochs: int,
             s_val: (DataLoader, Metric, Callable),
             args: Optional[Namespace] = None,
             device: str = 'cuda:0',
             epoch_offset: int = 0) -> Tuple[nn.Module, nn.Module, nn.Module]:
    torch.autograd.set_detect_anomaly(True)
    ckpt_dir = 'df_stat_ckpt.pt'
    logger.info(f'Teacher: {validate(teacher, criteria=s_val[1], loader=s_val[0], args=args, device=device)}')

    set_eval([teacher], device)
    set_train([student, decoder], device)
    to(list(optimizer), device)

    hooks = []
    for m in teacher.modules():
        if isinstance(m, nn.BatchNorm2d):
            hooks.append(FeatureMeanVarHook(m))
    args.hooks = hooks
    args.bn_mean = torch.cat([h.module.running_mean for h in hooks], dim=0).to(device)
    args.bn_var = torch.cat([h.module.running_var for h in hooks], dim=0).to(device)

    for c in criterion:
        c.reset()

    config = dataset_config[args.dataset]

    val_z = torch.randn(args.batch_size * 10, args.latent_size)

    val_loader = DataLoader(TensorDataset(val_z), shuffle=False, **args.loader_kwargs)
    best_result = None if epoch_offset == 0 else args.best_result
    for epoch in range(epoch_offset, n_epochs):
        bar = tqdm(range(200))
        for i in bar:
            set_train([student, encoder, decoder], device)
            # for _ in range(10):
            z = torch.randn(size=(args.batch_size, args.latent_size)).to(device)

            # Decoder
            img = decoder(z)
            n_img = normalize(img, *config.normalize)
            t_outputs = teacher(n_img)
            s_outputs = student(n_img)
            batch_mean = torch.cat([h.mean for h in args.hooks], dim=0)
            batch_var = torch.cat([h.var for h in args.hooks], dim=0)

            loss_d = criterion[0].update(t_outputs, s_outputs, batch_mean=batch_mean - args.bn_mean,
                                         batch_var=batch_var - args.bn_var).get_results()

            optimizer[0].zero_grad()
            loss_d.backward()
            optimizer[0].step()

            bar.set_description('Epoch: {:04d} {} {} {}'.format(epoch, criterion[0], criterion[1], criterion[2]))
            criterion[0].reset()

            # Student and Encoder
            for _ in range(5):
                z = torch.randn(size=(args.batch_size, args.latent_size)).to(device)
                with torch.no_grad():
                    img = decoder(z)
                    n_img = normalize(img, *config.normalize).detach()
                    t_outputs = teacher(n_img)
                zz, mu, log_var = encoder(img, labels=t_outputs.argmax(dim=1), return_dist=True)
                s_outputs = student(n_img)

                loss_s = criterion[1].update(s_outputs, t_outputs).get_results()
                optimizer[1].zero_grad()
                loss_s.backward()
                optimizer[1].step()

                loss_e = criterion[2].update(zz, z, mu=mu, log_var=log_var).get_results()
                optimizer[2].zero_grad()
                loss_e.backward()
                optimizer[2].step()

                bar.set_description(
                    'Epoch: {:04d} {} {} {}'.format(epoch, criterion[0], criterion[1], criterion[2]))
                criterion[1].reset()
                criterion[2].reset()
                del z

        scheduler[0].step()
        scheduler[1].step()

        # Validation Stage
        set_eval([student, encoder, decoder], device)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = tuple(t.to(device) for t in batch)
                z = batch[0]

                img = decoder(z)
                n_img = normalize(img, *config.normalize)
                t_outputs = teacher(n_img)
                s_outputs = student(n_img)
                batch_mean = torch.cat([h.mean for h in args.hooks], dim=0)
                batch_var = torch.cat([h.var for h in args.hooks], dim=0)
                zz, mu, log_var = encoder(img.detach(), labels=t_outputs.argmax(dim=1), return_dist=True)

                criterion[0].update(t_outputs, s_outputs, batch_mean=batch_mean - args.bn_mean,
                                    batch_var=batch_var - args.bn_var)
                criterion[2].update(zz, z, mu=mu, log_var=log_var)

            eval_metrics = validate(student, criteria=s_val[1], loader=s_val[0], args=args, device=device,
                                    show_progress=False)
            result = criterion[0].get_results(), criterion[2].get_results(), eval_metrics.get_results()
            logger.info(
                f'Eval [{epoch}/{n_epochs}]: Decoder Loss: {result[0]:8.3f} Encoder Loss: {result[1]:8.3f} {eval_metrics}')
            eval_metrics.reset()
            s_val[1].reset()
            criterion[0].reset()
            criterion[2].reset()

        if best_result is None or \
            result[0] < best_result[0] or \
            (result[0] == best_result[0] and result[1] < best_result[1]) or \
            (result[0] == best_result[0] and result[1] == best_result[1] and s_val[2](result[2], best_result[2])):
            logger.info(f'Best is saved at epoch {epoch}')
            best_result = result
            res = {
                'epoch': epoch,
                'optim': [o.state_dict() for o in optimizer],
                'sched': [s.state_dict() for s in scheduler],
                'best_result': best_result,
                'cur_stage': Stage.Df,
                'decoder': decoder.state_dict(),
                'encoder': encoder.state_dict(),
                'student': student.state_dict(),
            }

            torch.save(res, os.path.join(args.log_dir, ckpt_dir))

    try:
        ckpt = torch.load(os.path.join(args.log_dir, ckpt_dir))
        ckpt['epoch'] = n_epochs
        decoder.load_state_dict(ckpt['decoder'])
        student.load_state_dict(ckpt['student'])
        encoder.load_state_dict(ckpt['encoder'])
        torch.save(ckpt, os.path.join(args.log_dir, ckpt_dir))
    except FileNotFoundError:
        pass

    return decoder, student, encoder
