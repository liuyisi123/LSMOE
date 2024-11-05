#This code is a simplified version of the paper (LSM2OE). We will provide the complete project after the paper is officially released.
# import torch
# import torch.nn as nn  # 导入PyTorch的神经网络模块
# from torch.distributions.normal import Normal  # 导入正态分布模块
# import numpy as np  # 导入NumPy，一个用于科学计算的Python库
# import torch.nn.functional as F

# # 定义SparseDispatcher类，它是一个辅助类，用于实现混合专家系统
# class SparseDispatcher(object):
#     '''SparseDispatcher类用于创建专家的输入小批量，并将专家的结果组合成一个统一的输出张量。
#     它有两个主要方法：
#     dispatch - 接收一个输入张量，为每个专家创建输入张量。
#     combine - 接收每个专家的输出张量，并形成一个组合的输出张量。对于同一个批次元素，不同专家的输出将根据提供的“门控”权重进行求和。
#     该类使用一个“门控”张量进行初始化，该张量指定了哪些批次元素发送给哪些专家，以及在组合输出时使用的权重。如果门控张量在[b, e]位置不为0，则批次元素b被发送给专家e。
#     输入和输出都是二维的[batch, depth]。
#     调用者负责在调用此类之前折叠额外的维度，并在输出后将输出重塑为原始形状。
#     例如使用：
#     gates: 一个float32类型的`Tensor`，形状为`[batch_size, num_experts]`
#     inputs: 一个float32类型的`Tensor`，形状为`[batch_size, input_size]`
#     experts: 一个包含子网络的列表，长度为`num_experts`。
#     dispatcher = SparseDispatcher(num_experts, gates)
#     expert_inputs = dispatcher.dispatch(inputs)
#     expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
#     outputs = dispatcher.combine(expert_outputs)
#     上述代码设置了特定示例b的输出为：
#     output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
#     这个类利用门控矩阵中的稀疏性，只包括对专家i的`Tensor`中的批次元素，这些元素满足`gates[b, i] > 0`。
#     '''

#     def __init__(self, num_experts, gates):
#         '''创建一个SparseDispatcher实例。'''
#         # 门控张量，表示每个专家对应的权重
#         self._gates = gates
#         # 专家数量
#         self._num_experts = num_experts
#         # 对门控张量非零元素进行排序，并获取对应的索引
#         sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
#         # 获取专家的索引
#         _, self._expert_index = sorted_experts.split(1, dim=1)
#         # 获取每个专家对应的批次索引
#         self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
#         # 计算每个专家得到的样本数量
#         self._part_sizes = (gates > 0).sum(0).tolist()
#         # 扩展门控张量以匹配_batch_index，并使用gather函数重新排列非零门控值
#         gates_exp = gates[self._batch_index.flatten()]
#         self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

#     def dispatch(self, inp):
#         '''为每个专家创建一个输入张量。
#         专家`i`的`Tensor`包含了`inp`中对应批次元素`b`的切片，其中`gates[b, i] > 0`。
#         参数:
#           inp: 一个形状为"[batch_size, <extra_input_dims>]"的`Tensor`
#         返回:
#           一个包含`num_experts`个`Tensor`的列表，每个形状为`[expert_batch_size_i, <extra_input_dims>]`。
#          */'''
#         # 将样本分配给门控值非零的专家

#         # 根据批次索引扩展，之后可以根据_part_sizes进行分割
#         inp_exp = inp[self._batch_index].squeeze(1)
#         # 根据每个专家得到的样本数量分割输入
#         return torch.split(inp_exp, self._part_sizes, dim=0)

#     def combine(self, expert_out, multiply_by_gates=True):
#         '''将专家的输出按门控权重求和。
#         特定批次元素`b`的切片是通过对所有专家`i`的专家输出进行求和来计算的，由相应的门控值进行加权。如果`multiply_by_gates`设置为False，则忽略门控值。
#         参数:
#           expert_out: 一个包含`num_experts`个`Tensor`的列表，每个形状为`[expert_batch_size_i, <extra_output_dims>]`。
#           multiply_by_gates: 一个布尔值
#         返回:
#           一个形状为`[batch_size, <extra_output_dims>]`的`Tensor`。
#          */'''
#         # 对专家输出应用指数函数，使我们不再处于对数空间
#         stitched = torch.cat(expert_out, 0)

#         if multiply_by_gates:
#             # 如果需要按门控权重乘以非零门控值
#             stitched = stitched.mul(self._nonzero_gates)
#         # 创建一个零张量，其大小与门控张量相同，用于组合输出
#         zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
#         # 组合由相同k个专家处理的样本
#         combined = zeros.index_add(0, self._batch_index, stitched.float())
#         return combined

#     def expert_to_gates(self):
#         '''获取每个专家对应的门控值。
#         返回:
#           一个包含`num_experts`个一维`Tensor`的列表，类型为`tf.float32`，形状为`[expert_batch_size_i]`
#          */
#          '''
#         # 为每个专家分割非零门控值
#         return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

# # 定义一个简单的多层感知器类，作为专家网络
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
#         # 第一个卷积层
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         # 第二个卷积层
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)

#         # 残差连接的卷积层
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != self.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(self.expansion * out_channels)
#             )

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(identity)
#         out = F.relu(out)
#         return out

# class ResNetExpert(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(ResNet34Expert, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv1d(input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
#         self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
#         self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
#         self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool1d((1))
#         self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

#     def _make_layer(self, block, out_channels, blocks, stride):
#         layers = [block(self.in_channels, out_channels, stride)]
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         print(x.shape)
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         print(x.shape)
#         return out


# class MoE(nn.Module):
#     '''定义一个稀疏门控混合专家层，使用一层前馈网络作为专家。
#     参数:
#         input_size: 输入尺寸的整数
#         output_size: 输出尺寸的整数
#         num_experts: 专家数量的整数
#         hidden_size: 专家隐藏层大小的整数
#         noisy_gating: 是否使用带噪声的门控，布尔值
#         k: 每个批次元素使用的专家数量的整数
#      */
#      '''

#     def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
#         super(MoE, self).__init__()
#         # 假设输入尺寸和隐藏层尺寸对应于卷积网络的通道数
#         self.experts = nn.ModuleList([ResNet34Expert(input_size, hidden_size) for _ in range(num_experts)])
#         # 初始化参数
#         self.noisy_gating = noisy_gating  # 是否使用带噪声的门控
#         self.num_experts = num_experts  # 专家数量
#         self.output_size = output_size    # 输出尺寸
#         self.input_size = input_size      # 输入尺寸
#         self.hidden_size = hidden_size    # 隐藏层大小
#         self.k = k                        # 每个批次元素使用的专家数量

#         # 实例化专家网络，使用ModuleList来自动注册子模块
#         # self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
#         # self.experts = nn.ModuleList([ResNet34Expert(BasicBlock, [3, 4, 6, 3], input_size, hidden_size) for i in range(num_experts)])

#         # 门控网络的权重参数，初始化为零
#         self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
#         self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

#         # Softplus激活函数，用于噪声标准差的正则化
#         self.softplus = nn.Softplus()
#         # Softmax激活函数，用于门控输出的归一化
#         self.softmax = nn.Softmax(1)
#         # 注册常数作为缓冲区，不参与梯度计算
#         self.register_buffer("mean", torch.tensor([0.0]))
#         self.register_buffer("std", torch.tensor([1.0]))
#         # 断言k不大于专家数量
#         assert(self.k <= self.num_experts)

#     def cv_squared(self, x):
#         '''计算样本的方差系数的平方。
#         用作损失函数，以鼓励正分布更加均匀。
#         添加了一个小常数ε以保证数值稳定性。
#         如果Tensor为空，则返回0。
#         参数:
#             x: 一个`Tensor`。
#         返回:
#             a `Scalar`。
#          */
#          '''
#         eps = 1e-10  # 小常数ε
#         if x.shape[0] == 1:  # 如果只有一个专家，则返回0
#             return torch.tensor([0], device=x.device, dtype=x.dtype)
#         # 计算方差系数的平方
#         return x.float().var() / (x.float().mean()**2 + eps)

#     def _gates_to_load(self, gates):
#         '''计算给定门控下每个专家的真实负载。
#         负载是指对应门控大于0的样本数量。
#         参数:
#             gates: 形状为[batch_size, n]的`Tensor`。
#         返回:
#             形状为[n]的float32 `Tensor`。
#          */
#          '''
#         return (gates > 0).sum(0)  # 计算门控大于0的样本数量

#     def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
#         '''辅助NoisyTopKGating的函数。
#         给定不同的随机噪声，计算值进入前k名的概率。
#         这为我们提供了一种从损失函数反向传播的方法，该损失函数平衡了每个专家在每个示例中的前k名专家的次数。
#         如果没有噪声，传入noise_stddev为None，结果将不可微。
#         参数:
#             clean_values: 形状为[batch, n]的`Tensor`。
#             noisy_values: 形状为[batch, n]的`Tensor`。等于清洁值加上标准差为noise_stddev的正态分布噪声。
#             noise_stddev: 形状为[batch, n]的`Tensor`，或None。
#             noisy_top_values: 形状为[batch, m]的`Tensor`。
#                 "values"是tf.top_k(noisy_top_values, m)的输出。m >= k+1
#         返回:
#             形状为[batch, n]的`Tensor`。
#          */
#          '''
#         batch = clean_values.size(0)
#         m = noisy_top_values.size(1)
#         top_values_flat = noisy_top_values.flatten()

#         # 计算阈值位置
#         threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
#         threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
#         is_in = torch.gt(noisy_values, threshold_if_in)
#         threshold_positions_if_out = threshold_positions_if_in - 1
#         threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

#         # 计算每个值目前在前k名的概率
#         normal = Normal(self.mean, self.std)
#         prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
#         prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
#         prob = torch.where(is_in, prob_if_in, prob_if_out)
#         return prob

#     def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
#         '''带噪声的top-k门控。
#         参见论文: https://arxiv.org/abs/1701.06538
#         参数:
#             x: 输入张量，形状为[batch_size, input_size]
#             train: 一个布尔值 - 我们只在训练时添加噪声。
#             noise_epsilon: 一个小常数
#         返回:
#             gates: 形状为[batch_size, num_experts]的张量
#             load: 形状为[num_experts]的张量
#          */
#          '''
#         clean_logits = x @ self.w_gate  # 清洁的门控逻辑值
#         if self.noisy_gating and train:
#             # 训练时添加噪声
#             raw_noise_stddev = x @ self.w_noise
#             noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
#             noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
#             logits = noisy_logits
#         else:
#             logits = clean_logits

#         # 计算topk + 1，用于带噪声的门控
#         logits = self.softmax(logits)
#         top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
#         top_k_logits = top_logits[:, :self.k]
#         top_k_indices = top_indices[:, :self.k]
#         # 归一化处理
#         top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

#         zeros = torch.zeros_like(logits, requires_grad=True)
#         gates = zeros.scatter(1, top_k_indices, top_k_gates)

#         if self.noisy_gating and self.k < self.num_experts and train:
#             # 计算负载
#             load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
#         else:
#             load = self._gates_to_load(gates)
#         return gates, load

#     def forward(self, x, loss_coef=1e-2):
#         '''定义MoE层的前向传播。
#         参数:
#             x: 输入张量，形状为[batch_size, input_size]
#             train: 一个布尔标量。
#             loss_coef: 标量 - 负载平衡损失的乘数

#         返回:
#             y: 输出张量，形状为[batch_size, output_size]。
#             extra_training_loss: 标量。这应该被添加到模型的总训练损失中。这个损失的反向传播
#             鼓励所有专家在批次中被大致平等地使用。
#          */
#          '''
#         gates, load = self.noisy_top_k_gating(x, self.training)  # 计算门控和负载
#         # 计算重要性损失
#         importance = gates.sum(0)
#         # 计算负载损失和重要性损失
#         loss = self.cv_squared(importance) + self.cv_squared(load)
#         loss *= loss_coef

#         # 创建SparseDispatcher实例，并将输入分配给专家
#         dispatcher = SparseDispatcher(self.num_experts, gates)
#         expert_inputs = dispatcher.dispatch(x)
#         gates = dispatcher.expert_to_gates()
#         # 计算每个专家的输出
#         expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
#         y = dispatcher.combine(expert_outputs)  # 组合专家的输出
#         return y, loss


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined
    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

from torch.distributions.normal import Normal

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, output_size, hidden_channels, num_blocks=20):
        super(ResNet, self).__init__()
        self.in_channels = hidden_channels
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(hidden_channels, hidden_channels, num_blocks)
        self.fc_out = nn.Linear(hidden_channels, output_size)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = x.mean(dim=-1)
        x = self.fc_out(x)
        return x

class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_channels, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.k = k
        self.experts = nn.ModuleList([ResNet(1, output_size, hidden_channels) for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_inputs = [inp.unsqueeze(1) for inp in expert_inputs]  # Adding channel dimension
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

