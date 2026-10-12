import torch
from torch import nn


# RMSNorm: y_(i,j) = weight_j * (x_(i,j) / sqrt(sum(x_(i,:)^2)/d + eps))
# 假设 x 为 (b*s, hidden_size), b*s 为 token 个数， hidden_size 为特征维度
# b*s 就是 i 维度， hidden_size 就是 j 维度
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.square() 对全部 x 元素求 ^2 
        # .mean 对最后一个维度其和并平均
        mean_square = x.square().mean(dim=-1, keepdim=True)
        # 求出 (x_(i,j) / sqrt(sum(x_(i,:)^2)/d + eps))
        normalized = x * torch.rsqrt(mean_square + self.eps)

        output = normalized * self.weight

        return output


x = torch.randn(2, 16, 128, requires_grad=True)

norm = RMSNorm(dim=128, eps=1e-6)
reference = nn.RMSNorm(128, eps=1e-6)

# 使用非全一权重验证缩放计算。
with torch.no_grad():
    norm.weight.uniform_(0.5, 1.5)
    reference.weight.copy_(norm.weight)

y = norm(x)  # shape: (2, 16, 128)

torch.testing.assert_close(
    y, reference(x), rtol=1e-5, atol=1e-6
)

print("PASS")

y.square().mean().backward()