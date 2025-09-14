# import numpy as np
# import torch


# def DataTransform(sample, config):

#     weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
#     strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

#     return weak_aug, strong_aug


# def jitter(x, sigma=0.8):
#     # https://arxiv.org/pdf/1706.00527.pdf
#     return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


# def scaling(x, sigma=1.1):
#     # https://arxiv.org/pdf/1706.00527.pdf
#     factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
#     ai = []
#     for i in range(x.shape[1]):
#         xi = x[:, i, :]
#         ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
#     return np.concatenate((ai), axis=1)


# def permutation(x, max_segments=5, seg_mode="random"):
#     orig_steps = np.arange(x.shape[2])

#     num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

#     ret = np.zeros_like(x)
#     for i, pat in enumerate(x):
#         if num_segs[i] > 1:
#             if seg_mode == "random":
#                 split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
#                 split_points.sort()
#                 splits = np.split(orig_steps, split_points)
#             else:
#                 splits = np.array_split(orig_steps, num_segs[i])
#             warp = np.concatenate(np.random.permutation(splits)).ravel()
#             ret[i] = pat[0,warp]
#         else:
#             ret[i] = pat
#     return torch.from_numpy(ret)

import torch

def DataTransform(sample, config):
    # 统一为 torch.Tensor
    x = sample if isinstance(sample, torch.Tensor) else torch.as_tensor(sample)
    weak_aug = scaling(x, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(x, max_segments=config.augmentation.max_seg),
                        config.augmentation.jitter_ratio)
    return weak_aug, strong_aug

def jitter(x, sigma=0.8):
    # 按元素加高斯噪声
    return x + torch.randn_like(x) * sigma

def scaling(x, sigma=1.1):
    # 保持与原逻辑一致：mean=2.0（如需更温和可改为 1.0）
    B, C, T = x.shape
    factor = torch.normal(mean=2.0, std=sigma, size=(B, 1, T), device=x.device, dtype=x.dtype)
    return x * factor

def permutation(x, max_segments=5, seg_mode="random"):
    # 对所有通道一起按相同时间重排；不要求等长分段
    assert x.dim() == 3, f"expect [B,C,T], got {x.shape}"
    B, C, T = x.shape
    out = []
    for b in range(B):
        k = int(torch.randint(1, max_segments + 1, (1,), device=x.device).item())
        if k <= 1 or T <= 1:
            out.append(x[b])
            continue

        if seg_mode == "random":
            # 随机切分：在 1..T-1 里取 k-1 个切点
            if (k - 1) > 0 and (T - 1) > 0:
                splits = torch.sort(torch.randperm(T - 1, device=x.device)[:k - 1] + 1).values
            else:
                splits = torch.tensor([], device=x.device, dtype=torch.long)
            bounds = torch.cat([
                torch.tensor([0], device=x.device),
                splits,
                torch.tensor([T], device=x.device)
            ])
        else:
            # 近似等长分段（允许不整除）
            bounds = torch.linspace(0, T, steps=k + 1, device=x.device).round().long()
            bounds[0] = 0
            bounds[-1] = T

        order = torch.randperm(k, device=x.device)
        chunks = [x[b, :, bounds[i]:bounds[i + 1]] for i in range(k)]
        out.append(torch.cat([chunks[j] for j in order], dim=-1))

    return torch.stack(out, dim=0)