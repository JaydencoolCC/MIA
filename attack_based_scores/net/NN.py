from torch import nn
from itertools import repeat


# class SpatialDropout(nn.Module):
#     """
#     空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
#     如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
#     若沿着axis=2则可对某些token进行整体dropout
#     """
#
#     def __init__(self, drop=0.5):
#         super(SpatialDropout, self).__init__()
#         self.drop = drop
#
#     def forward(self, inputs, noise_shape=None):
#         """
#         @param: inputs, tensor
#         @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
#         """
#         outputs = inputs.clone()
#         if noise_shape is None:
#             noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape
#
#         self.noise_shape = noise_shape
#         if not self.training or self.drop == 0:
#             return inputs
#         else:
#             noises = self._make_noises(inputs)
#             if self.drop == 1:
#                 noises.fill_(0.0)
#             else:
#                 noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
#             noises = noises.expand_as(inputs)
#             outputs.mul_(noises)
#             return outputs
#
#     def _make_noises(self, inputs):
#         return inputs.new().resize_(self.noise_shape)


class NN_Model(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):

        super(NN_Model, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=n_in[1],
                      out_features=n_hidden),
            nn.Tanh()
        )

        # self.d1 = SpatialDropout(drop=0.5)
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=n_hidden,
                      out_features=n_out)
        )

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)

        return x
