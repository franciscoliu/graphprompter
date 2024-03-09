from typing import Iterable
import math
import sys
import os
import datetime
import json
import time
import torch
import gc
import numpy as np
import wandb

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import timm.optim.optim_factory as optim_factory

from torch.utils.tensorboard import SummaryWriter


from src.model import load_model, llama_model_path
from src.dataset import load_dataset
import src.utils.misc as misc
from src.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from src.config import parse_args_llama
from src.utils.lr_schedule import adjust_learning_rate
from src.utils.seed import seed_everything

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    accum_iter = args.grad_steps

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        c_loss = model(batch)
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.grad_steps

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        with torch.no_grad():
            c_loss = model(batch)
        loss = c_loss
        loss_value = loss.item()

        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.grad_steps

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        with torch.no_grad():
            preds = model.inference(batch)
            print(preds)

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    seed = args.seed
    wandb.init(project=f"{args.project}",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args)

    seed_everything(seed=args.seed)
    print(args)

    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    sampler_train = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    sampler_val = torch.utils.data.DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler_val,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    data_loader_test = DataLoader(
        test_dataset,
        sampler=sampler_val,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph=dataset.graph, graph_type=dataset.graph_type, args=args)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.grad_steps * misc.get_world_size()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.wd)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    print(f"Start training for {args.num_epochs} epochs")
    start_time = time.time()
    for epoch in range(args.num_epochs):
        data_loader_train.sampler.set_epoch(epoch)
        data_loader_val.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args)
        val_stats = val_one_epoch(model, data_loader_val, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args)

        if args.output_dir and (epoch % 8 == 0 or epoch + 1 == args.num_epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            **{f"val_{k}": v for k, v in val_stats.items()},
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    
    test_one_epoch(model, data_loader_test, optimizer, device, 0, loss_scaler, log_writer=log_writer, args=args)


if __name__ == "__main__":

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
