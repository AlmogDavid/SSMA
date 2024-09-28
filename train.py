import os
import random
from typing import Optional, Tuple

import hydra
import numpy as np
import torch
import torch_geometric
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn, optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv
from torch_geometric.transforms import AddRandomWalkPE
from tqdm import tqdm

from data.datasets import BaseDataset
from models.pyg_net import PYGMPNet
from ssma import SSMA
from utils import ResultsLogger


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def load_ds(cfg: DictConfig) -> BaseDataset:
    pre_transform = None
    if cfg.layer_type == "graphgps" and cfg.use_pe_gps:
        pre_transform = AddRandomWalkPE(walk_length=cfg.rw_pe_depth_gps, attr_name='pe')
    return BaseDataset.get_all_ds()[cfg.dataset](pre_transform=pre_transform)


def create_dl(cfg: DictConfig, ds: BaseDataset, k: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds.set_current_fold(k)
    train_ds = ds.get_split("train")
    valid_ds = ds.get_split("valid")
    test_ds = ds.get_split("test")

    if cfg.debug: # Take subset
        train_ds = train_ds[:128]
        valid_ds = valid_ds[:128]
        test_ds = test_ds[:128]

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_dl, valid_dl, test_dl


def get_num_trainable_params(model: nn.Module) -> int:
    num_trainable_params = 0
    for name, param in model.named_parameters():
        try:
            if param.requires_grad:
                num_trainable_params += param.numel()
        except ValueError:
            print(f"Failed to compute number of parameters for parameter named {name}")
            raise
    return num_trainable_params


def create_model(cfg: DictConfig, ds: BaseDataset, ds_sample: Data, device: str) -> nn.Module:
    available_models = PYGMPNet.get_all_models()
    ds_sample = ds_sample.cpu()
    new_edge_index, new_edge_attr = torch_geometric.utils.subgraph(subset=list(range(min(ds_sample.x.size(0), 32))),
                                                       edge_index=ds_sample.edge_index,
                                                       edge_attr=ds_sample.edge_attr,
                                                       relabel_nodes=True)
    data_kwargs = {"x": ds_sample.x[:32], "edge_index": new_edge_index, "y": ds_sample.y[:32],
                   "edge_attr": new_edge_attr}
    if hasattr(ds_sample, "pe"):
        data_kwargs["pe"] = ds_sample.pe[:32]

    ds_sample = Data(**data_kwargs)
    ds_sample = Batch.from_data_list([ds_sample])

    if cfg.parameter_budget > 0:
        print("Searching for highest hidden dimension with the parameter budget")
        low = 1
        high = 2048
        mid = 0
        while low < high:
            new_mid = (low + high) // 2

            if cfg.layer_type == "pna": # Dim should be divisible by number of towers
                new_mid = new_mid  - new_mid % cfg.towers_pna
            elif cfg.layer_type == "gat": # Dim should be divisible by number of heads
                new_mid = new_mid  - new_mid % cfg.n_heads_gat
            elif cfg.layer_type == "gat2": # Dim should be divisible by number of heads
                new_mid = new_mid - new_mid % cfg.n_heads_gat2
            elif cfg.layer_type == "graphgps":
                new_mid = new_mid - new_mid % cfg.n_heads_gps

            if new_mid == mid: # No progress
                break
            mid = new_mid

            cfg.hidden_dim = mid
            model = available_models[cfg.model](cfg=cfg, ds=ds)
            model(ds_sample) # Initialize model parameters defined as lazy modules
            num_params = get_num_trainable_params(model)
            print(f"Trying width: {mid}, num params: {num_params}")
            if num_params > cfg.parameter_budget:
                high = mid
            else:
                low = mid + 1
        print(f"Selected model width: {mid}")
    else:
        hidden_dim = cfg.hidden_dim
        if cfg.layer_type == "pna":  # Dim should be divisible by number of towers
            new_hidden = hidden_dim - hidden_dim % cfg.towers_pna
        elif cfg.layer_type == "gat":  # Dim should be divisible by number of heads
            new_hidden = hidden_dim - hidden_dim % cfg.n_heads_gat
        elif cfg.layer_type == "gat2":  # Dim should be divisible by number of heads
            new_hidden = hidden_dim - hidden_dim % cfg.n_heads_gat2
        elif cfg.layer_type == "graphgps":
            new_hidden = hidden_dim - hidden_dim % cfg.n_heads_gps
        else:
            new_hidden = hidden_dim

        if new_hidden != hidden_dim:
            print(f"Warning: Hidden dimension {hidden_dim} not divisible by number of heads/towers. Setting to {new_hidden}")
            cfg.hidden_dim = new_hidden

        model = available_models[cfg.model](cfg=cfg, ds=ds)
        model(ds_sample)  # Initialize model parameters defined as lazy modules

    model = model.to(device)
    return model


@torch.no_grad()
def validate(model: nn.Module, ds: BaseDataset, valid_dl: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    agg_pred = []
    agg_gt = []
    for data in valid_dl:
        data = data.to(device)
        output = model(data)

        agg_pred.append(output.detach().cpu())
        agg_gt.append(data.y.detach().cpu())

    agg_pred = torch.cat(agg_pred, dim=0)
    agg_gt = torch.cat(agg_gt, dim=0)
    epoch_loss = ds.compute_loss(pred=agg_pred, gt=agg_gt)
    epoch_metric = ds.compute_metric(pred=agg_pred, gt=agg_gt)
    return epoch_loss.item(), epoch_metric.item()


def train_model(cfg: DictConfig, model: nn.Module, ds: BaseDataset, train_dl: DataLoader, valid_dl: DataLoader, test_dl: DataLoader):
    metric_name = ds.metric_name()
    metric_type = ds.metric_type()

    device = get_device()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.init_lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode="min",
                                                     factor=cfg.lr_reduce_factor,
                                                     patience=cfg.lr_schedule_patience)

    best_train_loss = best_val_loss = best_test_loss = None
    best_metric_train = best_metric_val = best_metric_test = None

    try:
        train_iter = iter(train_dl)
        agg_pred = []
        agg_gt = []
        eval_every = min(len(train_dl), cfg.eval_every)
        print(f"Evaluating model every {eval_every} steps")
        step = 0
        with tqdm(desc="Training") as t:
            while True:
                try:
                    sample = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dl)
                    sample = next(train_iter)

                model.train()
                sample = sample.to(device)
                optimizer.zero_grad()
                output = model(sample)
                loss = ds.compute_loss(pred=output, gt=sample.y)
                loss.backward()
                optimizer.step()

                loss_cpu = loss.detach().cpu()
                output_cpu = output.detach().cpu()
                gt_cpu = sample.y.detach().cpu()
                agg_pred.append(output_cpu)
                agg_gt.append(gt_cpu)

                if np.isnan(loss_cpu).any():
                    print(f'Step {step}: Nan detected for training loss.')
                    break

                if step % eval_every == eval_every - 1:
                    model.eval()

                    agg_pred = torch.cat(agg_pred, dim=0)
                    agg_gt = torch.cat(agg_gt, dim=0)
                    curr_train_loss = ds.compute_loss(pred=agg_pred, gt=agg_gt).item()
                    curr_train_metric = ds.compute_metric(pred=agg_pred, gt=agg_gt).item()
                    agg_pred = []
                    agg_gt = []

                    curr_val_loss, curr_val_metric = validate(model=model, valid_dl=valid_dl, ds=ds, device=device)
                    curr_test_loss, curr_test_metric = validate(model=model, valid_dl=test_dl, ds=ds, device=device)

                    new_best = False
                    if best_train_loss is None:
                        new_best = True
                    else:
                        if metric_type == "minimize":
                            if curr_val_metric <= best_metric_val:
                                new_best = True
                        elif metric_type == "maximize":
                            if curr_val_metric >= best_metric_val:
                                new_best = True
                        else:
                            raise RuntimeError(f"Unknown metric type: {ds.metric_type()}")

                    if new_best:
                        best_train_loss = curr_train_loss
                        best_val_loss = curr_val_loss
                        best_test_loss = curr_test_loss
                        best_metric_train = curr_train_metric
                        best_metric_val = curr_val_metric
                        best_metric_test = curr_test_metric

                    log_dict = dict(
                        lr=optimizer.param_groups[0]['lr'],
                        train_loss=curr_train_loss,
                        val_loss=curr_val_loss,
                        test_loss=curr_test_loss,
                        best_train_loss=best_train_loss,
                        best_val_loss=best_val_loss,
                        best_test_loss=best_test_loss)
                    log_dict[metric_name + "_train"] = curr_train_metric
                    log_dict[metric_name + "_val"] = curr_val_metric
                    log_dict[metric_name + "_test"] = curr_test_metric
                    log_dict["best_" + metric_name + "_train"] = best_metric_train
                    log_dict["best_" + metric_name + "_val"] = best_metric_val
                    log_dict["best_" + metric_name + "_test"] = best_metric_test
                    ResultsLogger.get_instance().log(**log_dict)

                    tqdm_log_dict = dict(lr=optimizer.param_groups[0]['lr'],
                                         train_loss=curr_train_loss, val_loss=curr_val_loss, test_loss=curr_test_loss)
                    tqdm_log_dict[f"{metric_name}_train"] = curr_train_metric
                    tqdm_log_dict[f"{metric_name}_val"] = curr_val_metric
                    tqdm_log_dict[f"{metric_name}_test"] = curr_test_metric
                    t.set_postfix(**tqdm_log_dict)

                    scheduler.step(curr_val_loss)
                    if optimizer.param_groups[0]['lr'] < cfg.min_lr:
                        print("\n!! LR EQUAL TO MIN LR SET.")
                        break

                step += 1
                t.update(1)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        print('-' * 89)

    print("Training finished")
    if best_train_loss is not None: # None means training was interrupted on first step
        print(f"Best train loss: {best_train_loss:.4f}")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Best test loss: {best_test_loss:.4f}")
        print(f"Best train {metric_name}: {best_metric_train:.4f}")
        print(f"Best val {metric_name}: {best_metric_val:.4f}")
        print(f"Best test {metric_name}: {best_metric_test:.4f}")


@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(0)
        torch.backends.cudnn.benchmark = False

    # Pretty print the configuration
    print(OmegaConf.to_yaml(cfg, resolve=True))

    runs = cfg.runs if cfg.k_fold <= 0 else cfg.k_fold
    use_k_fold = cfg.k_fold > 0
    use_ssma = cfg.aggr == "SSMA" or "SSMA" in cfg.aggr

    job_name = HydraConfig.get().job.config_name
    exp_name = job_name + f"_{cfg.layer_type}"
    exp_name = exp_name + "_SSMA" if use_ssma else exp_name

    # Load ds
    ds = load_ds(cfg)
    cfg.use_edge_feat = cfg.use_edge_feat and ds.edge_dim() > 0

    with ResultsLogger(exp_name=exp_name) as res_logger:
        for run_idx in range(runs):
            k_fold = run_idx if use_k_fold else None
            train_loader, valid_loader, test_loader = create_dl(cfg, ds, k_fold)

            if run_idx == 0 and cfg.layer_type == "pna": # Compute PNA statistics
                print("Computing PNA statistics")
                deg_hist_pna = PNAConv.get_degree_histogram(train_loader)
                with open_dict(cfg):
                    cfg.avg_d_pna = deg_hist_pna.numpy().tolist()

            device = get_device()
            data_sample = next(iter(train_loader)).to(device)
            model = create_model(cfg=cfg, ds=ds, ds_sample=data_sample, device=device)

            if run_idx == 0:
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                cfg_dict["num_params"] = get_num_trainable_params(model)
                print("Number of trainable parameters: ", cfg_dict["num_params"])
                res_logger.setup(config=cfg_dict)

            res_logger.mark_new_run()
            train_model(cfg=cfg, model=model, ds=ds, train_dl=train_loader, valid_dl=valid_loader, test_dl=test_loader)

            if cfg.save_affine_mat and "SSMA" in cfg.aggr:
                target_dir = f"affine_mats/{exp_name}_{run_idx}"
                os.makedirs(target_dir, exist_ok=True)
                for layer_idx, layer in enumerate(model.layers):
                    aggr = layer.aggr_module
                    if not isinstance(aggr, SSMA):
                        print("Can't save affine mat for non-SSMA aggregationr")
                        continue
                    affine_mat = aggr._affine_layer.weight.detach().cpu().numpy()
                    affine_bias = aggr._affine_layer.bias.detach().cpu().numpy()
                    np.save(f"{target_dir}/affine_mat_{layer_idx}.npy", affine_mat)
                    np.save(f"{target_dir}/affine_bias_{layer_idx}.npy", affine_bias)

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
