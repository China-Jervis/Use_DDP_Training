# Pytorch官方教程

## 1.Pytorch分布式概述

### 1.1 简介

* 分布式数据并行训练（DDP）是一种广泛采用的单程序多数据的训练方式。 通过 DDP，模型会在每个进程上进行复制，并且每个模型副本都将被提供一组不同的输入数据样本。 DDP 负责梯度通信以保持模型副本同步，并将其与梯度计算结合以加速训练
* 基于 RPC 的分布式训练 (RPC) 支持无法适应数据并行训练的通用训练结构，例如分布式管道并行、参数服务器范例以及 DDP 与其他训练范例的组合。 它有助于管理远程对象的生命周期并将 autograd 引擎扩展到机器边界之外
* 集体通信（c10d）库支持在组内跨进程发送张量。 它提供集体通信 API（例如 all_reduce 和 all_gather）和 P2P 通信 API（例如 send 和 isend）。 DDP和RPC（ProcessGroup Backend）都是建立在c10d之上的，前者采用集体通信，后者采用P2P通信。 通常，开发人员不需要直接使用这种原始通信 API，因为 DDP 和 RPC API 可以服务于许多分布式训练场景。 但是，在某些用例中，此 API 仍然很有用。 一个例子是分布式参数平均，其中应用程序希望在向后传递之后计算所有模型参数的平均值，而不是使用 DDP 来传达梯度。 这可以将通信与计算解耦，并允许对通信内容进行更细粒度的控制，但另一方面，它也放弃了 DDP 提供的性能优化。 使用 PyTorch 编写分布式应用程序展示了使用 c10d 通信 API 的示例

### 1.2 数据并行训练

PyTorch 提供了多种数据并行训练方式。对于从简单到复杂、从原型到生产逐渐发展的应用程序，常见的开发轨迹是：

* 单GPU：如果数据和模型可以容纳在一个 GPU 中，并且对训练速度没有要求，可以使用单设备训练

* DP：使用单机多GPU DataParallel，在单机上利用多个GPU，以最少的代码更改来加速训练

* DDP：如果想进一步加快训练速度并愿意编写更多代码来设置DDP，可以使用单机多 GPU DistributedDataParallel

* 如果应用程序需要跨物理设备，可以使用DDP+启动脚本的方式

* 如果可能会出现错误（例如内存不足）或者资源可能在训练期间动态加入和离开，使用 torch.distributed.elastic 启动分布式训练

  *数据并行训练也可以使用自动混合精度 (AMP)*

#### 1.2.1 DP

DataParallel 包能够以最低的编码障碍实现单机多 GPU 并行性。 只需要对应用程序代码进行一行更改。 尽管 DataParallel 非常易于使用，但它通常无法提供最佳性能，因为它在每次前向传递中都会复制模型，并且其单进程多线程并行性会受到 GIL 争用的影响。 为了获得更好的性能，可以使用 DistributedDataParallel

#### 1.2.2 DDP

与DataParallel相比，DistributedDataParallel需要多一步设置，即调用init_process_group。DDP 使用多进程并行性，因此模型副本之间不存在 GIL 争用。 此外，模型在 DDP 构建时广播，而不是在每次前向传递中广播，这也有助于加快训练速度。DDP 附带了多种性能优化技术。

`torch.nn.parallel.DistributedDataParallel(DDP)`实现细节

* DDP 依赖 c10d ProcessGroup 进行通信。因此，应用程序必须在构建 DDP 之前创建 ProcessGroup 实例。`init_process_group()`
* 构建DDP：
  * DDP 构造函数引用本地模块，并将 state_dict() 从rank 0 广播到ProcessGroup中的所有其他进程，以确保所有模型副本从完全相同的状态开始
  * 每个DDP进程创建一个本地Reducer,之后它将在向后传递期间处理梯度同步，为了提高通信效率，Reducer将参数梯度组织成桶，一次减少一个桶。可以通过在 DDP 构造函数中设置bucket_cap_mb 参数来配置存储桶大小。从参数梯度到桶的映射是在构造时根据桶大小限制和参数大小确定的。模型参数按照给定模型中 Model.parameters() 的（大致）相反顺序分配到存储桶中，使用相反顺序的原因是因为 DDP 期望梯度在向后传递期间以该顺序准备好。除了分桶之外，Reducer 还在构造过程中注册 autograd 钩子，每个参数一个钩子。 当梯度准备好时，这些钩子将在向后传递过程中被触发。
  * **前向传播：**DDP 获取输入并将其传递到本地模型，然后如果 find_unused_parameters 设置为 True，则分析本地模型的输出。此模式允许在模型的子图上反向传播，DDP 通过从模型输出遍历 autograd 图并将所有未使用的参数标记进行标记，以此找出反向传播中涉及哪些参数。在反向传播过程中，Reducer只会等待未准备好的参数，但仍然会减少所有桶。目前，将参数梯度标记为就绪并不能帮助 DDP 跳过存储桶，但可以防止 DDP 在反向传播过程中永远等待不存在的梯度。遍历 autograd 图会带来额外的开销，因此应用程序应该仅在必要时将 find_unused_parameters 设置为 True。
  * **反向传播：**backward()函数直接在loss Tensor上调用，这不受DDP的控制，DDP使用在构造时注册的autograd hooks来触发梯度同步。 当一个梯度准备就绪时，该梯度累加器上相应的 DDP 挂钩将触发，然后 DDP 会将该参数梯度标记。 当一个存储桶中的梯度全部准备好时，Reducer 会在该存储桶上启动异步 allreduce，以计算所有进程的梯度平均值。 当所有bucket准备好后，Reducer将阻塞等待所有allreduce操作完成。 完成此操作后，平均梯度将写入所有参数的 param.grad 字段。 因此，在反向传播之后，不同DDP进程中相同对应参数的grad字段应该是相同的。
  * **更新梯度：**从优化器的角度来看，它正在优化局部模型。所有 DDP 进程上的模型副本都可以保持同步，因为它们都从相同的状态开始，并且在每次迭代中具有相同的平均梯度。

#### 1.2.3 DDP+启动脚本

 [Launching and configuring distributed data parallel applications](https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md) 

#### 1.2.4 ZeroRedundancyOptimizer

 [ZeroRedundancyOptimizer](https://pytorch.org/docs/stable/distributed.optim.html)减少优化器内存占用

#### 1.2.5 Context Manager

使用连接上下文管理器进行不均匀输入的分布式训练教程逐步介绍如何使用通用连接上下文进行不均匀输入的分布式训练。

### 1.3 torch.distributed.elastic

随着应用程序复杂性和规模的增长，故障恢复成为一项要求。 有时，使用 DDP 时不可避免地会遇到内存不足 (OOM) 等错误，但 DDP 本身无法从这些错误中恢复，并且无法使用标准的` try- except `结构来处理它们。 这是因为 DDP 要求所有进程以紧密同步的方式运行，并且不同进程中启动的所有 AllReduce 通信必须匹配。 如果组中的某个进程抛出异常，则可能会导致不同步（AllReduce 操作不匹配），从而导致崩溃或挂起。 `torch.distributed.elastic `增加了容错能力和利用动态机器池（弹性）的能力。

## 2. 分布式数据并行入门

### 2.1 API

该容器通过在每个模型副本之间同步梯度来提供数据并行性。要同步的设备由输入 `process_group `指定，默认情况下是整个世界。`DistributedDataParallel `不会在参与的 GPU 之间对输入数据进行分块或分片；用户负责定义如何执行此操作，例如通过使用 `DistributedSampler`。

```python
# 创建此类需要torch.distributed已通过调用
# torch.distributed.init_process_group()进行初始化
torch.nn.parallel.DistributedDataParallel(
  module,																	# 待并行化模块
  device_ids=None,												# 对于独占时，为int，非独占或CPU为None
  output_device=None,	
  dim=0, 
  broadcast_buffers=True, 								# 在广播功能开始时启用模块缓冲区同步的标志
  process_group=None, 										# 进程组
  bucket_cap_mb=25, 											# 存储桶大小
  find_unused_parameters=False, 					#寻找没有使用的参数
  check_reduction=False, 									# 已弃用参数
  gradient_as_bucket_view=False, 
  static_graph=False, 
  delay_all_reduce_named_params=None, 
  param_to_hook_all_reduce=None, 
  mixed_precision=None
)
```

要在具有 N 个 GPU 的主机上使用`DistributedDataParallel`，您应该生成 N 个进程，确保每个进程仅在 0 到 N-1 的单个 GPU 上工作。这可以通过为每个进程设置 CUDA_VISIBLE_DEVICES 或调用`torch.cuda.set_device(i)`完成

在每个进程中，参考以下内容来构建该模块：

```python
torch.distributed.init_process_group(
  backend='nccl',
  world_size=N,
  init_method='...'
)
model=DistributedDataParallel(model,device_ids=[i],output_device=i)
```

在每个节点生成多个进程，可以采用`torch.distributed.launch `	,`torch.multiprocessing.spawn`,`torchrun`

*note1:如果在一个进程上使用 torch.save 来保存模型，并在其他一些进程上使用 torch.load 来加载模型，请确保为每个进程正确配置了 map_location 。 如果没有map_location，torch.load会将模块恢复到保存该模块的设备。*

*note2:当模型在 Batch=N 的 M 个节点上训练时，如果跨实例的损失求和（而不是像往常一样平均），则与在 Batch=MxN 的单个节点上训练的相同模型相比，梯度将小 M 倍 批量大小（因为不同节点之间的梯度是平均的）。想要获得与本地训练在数学上等效的训练过程，应该考虑到这一点。 但在大多数情况下，可以将 DistributedDataParallel 包装的模型、DataParallel 包装模型和单个 GPU 上的普通模型视为相同（例如，对等效批量大小使用相同的学习率）*

*note3：模型参数永远不会在进程之间广播。 该模块对梯度执行全归约步骤，并假设优化器将在所有过程中以相同的方式修改它们。 在每次迭代中，缓冲区（例如 BatchNorm 统计信息）从处于 0 级进程的模块广播到系统中的所有其他副本。*