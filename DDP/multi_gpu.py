import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: 每个进程的全局id
        world_size: 全局进程数
    """
    # 设置通信地址以及通信接口/master节点
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # 初始化进程组，初始化进程组可以通过TCP或共享文件系统
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # 为每一个进程设置默认的GPU，这对于防止挂起或GPU:0内存使用率过高很重要
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id                # 当前运行的gpu_id
        self.model = model.to(gpu_id)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])            # 设置DDP model，设置当前的gpu_id

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)        # 划分数据集到每个GPU上
        for source, targets in self.train_loader:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()            # DDP model 需要获取其中module才能读取权重
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # 所有gpu上的模型权重相同，所以只需要保存一个就行
            if self.gpu_id == 0 and epoch % self.save_every == 0:   # SUOYOU
                self._save_checkpoint(epoch)


def load_train_objs():
    """设置dataset,model,optimizer"""
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """设置的dataloader，需要设置sampler参数"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    """运行主函数"""
    ddp_setup(rank, world_size)         # 设置DDP
    dataset, model, optimizer = load_train_objs()       # 加载数据集，模型，优化器
    train_loader = prepare_dataloader(dataset, batch_size)    # 设置训练dataloader
    trainer = Trainer(model, train_loader, optimizer, rank, save_every)   # 设置训练器
    trainer.train(total_epochs)             #开始训练
    destroy_process_group()              # 摧毁进程组


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=50, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=10, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)