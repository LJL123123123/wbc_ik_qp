import torch

def quat_to_rotmat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """支持 Batch 的四元数 (xyzw) 转旋转矩阵"""
    if q.dim() == 1:
        q = q.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    xx, yy, zz, ww = x*x, y*y, z*z, w*w
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w

    R = torch.stack([
        ww + xx - yy - zz, 2*(xy - zw),       2*(xz + yw),
        2*(xy + zw),       ww - xx + yy - zz, 2*(yz - xw),
        2*(xz - yw),       2*(yz + xw),       ww - xx - yy + zz
    ], dim=-1).reshape(-1, 3, 3)

    return R[0] if squeeze else R

def skew(w: torch.Tensor) -> torch.Tensor:
    """支持 Batch 的反对称矩阵生成"""
    if w.dim() == 1:
        wx, wy, wz = w
        return torch.tensor([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], device=w.device, dtype=w.dtype)
    
    B = w.shape[0]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    zeros = torch.zeros(B, device=w.device, dtype=w.dtype)
    
    # 构造 (B, 3, 3) 矩阵
    res = torch.stack([
        torch.stack([zeros, -wz,  wy], dim=-1),
        torch.stack([ wz, zeros, -wx], dim=-1),
        torch.stack([-wy,  wx, zeros], dim=-1)
    ], dim=-2)
    return res

def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """支持 Batch 的 SO(3) 指数映射"""
    if w.dim() == 1:
        w = w.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    theta = torch.linalg.norm(w, dim=-1, keepdim=True) # (B, 1)
    I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(w.shape[0], 3, 3)
    
    near_zero = (theta < 1e-8)
    
    # Rodrigues formula 处理
    W = skew(w / (theta + 1e-12))
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    
    R = I + sin_t * W + (1 - cos_t) * torch.bmm(W, W)
    
    # 接近 0 时返回一阶近似或单位阵
    res = torch.where(near_zero.unsqueeze(-1), I + skew(w), R)
    return res[0] if squeeze else res

def so3_left_jacobian(w: torch.Tensor) -> torch.Tensor:
    """支持 Batch 的左雅可比 V(w)"""
    if w.dim() == 1:
        w = w.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B = w.shape[0]
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)
    I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0).expand(B, 3, 3)
    W = skew(w)
    
    near_zero = (theta < 1e-8)
    theta2 = theta * theta
    
    A = (1 - torch.cos(theta)) / (theta2 + 1e-12)
    B_coeff = (theta - torch.sin(theta)) / (theta2 * theta + 1e-12)
    
    V = I + A.unsqueeze(-1) * W + B_coeff.unsqueeze(-1) * torch.bmm(W, W)
    
    # 接近 0 时使用级数近似
    V_small = I + 0.5 * W + (1.0/6.0) * torch.bmm(W, W)
    res = torch.where(near_zero.unsqueeze(-1), V_small, V)
    return res[0] if squeeze else res

def quat_mul_xyzw(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """支持 Batch 的四元数乘法"""
    if q.dim() == 1: q = q.unsqueeze(0)
    if r.dim() == 1: r = r.unsqueeze(0)
    
    x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    
    res = torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dim=-1)
    return res

def so3_exp_to_quat_xyzw(w: torch.Tensor) -> torch.Tensor:
    """支持 Batch 的旋转向量转四元数"""
    if w.dim() == 1:
        w = w.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    theta = torch.linalg.norm(w, dim=-1, keepdim=True)
    near_zero = (theta < 1e-8)
    
    axis = w / (theta + 1e-12)
    half = 0.5 * theta
    s = torch.sin(half)
    c = torch.cos(half)
    
    res = torch.cat([axis * s, c], dim=-1)
    # 接近 0 时返回 [0,0,0,1]
    default = torch.tensor([0., 0., 0., 1.], device=w.device, dtype=w.dtype).expand_as(res)
    final = torch.where(near_zero, default, res)
    return final[0] if squeeze else final

def integrate_freeflyer_quat_xyzw(q_fb: torch.Tensor,
                                 dp_body: torch.Tensor,
                                 domega_body: torch.Tensor,
                                 right_multiply: bool = True) -> torch.Tensor:
    """
    支持多 Batch 的自由浮动基座积分。
    q_fb: (B, 7) -> [px, py, pz, qx, qy, qz, qw]
    dp_body, domega_body: (B, 3) 增量
    """
    B = q_fb.shape[0]
    p = q_fb[:, 0:3]
    quat = q_fb[:, 3:7]

    R = quat_to_rotmat_xyzw(quat)
    R_delta = so3_exp(domega_body)
    V = so3_left_jacobian(domega_body)
    
    # 批量处理位移增量: (B, 3, 3) @ (B, 3, 1)
    t_delta_body = torch.bmm(V, dp_body.unsqueeze(-1)).squeeze(-1)

    if right_multiply:
        # p_next = p + R @ t_delta_body
        p_next = p + torch.bmm(R, t_delta_body.unsqueeze(-1)).squeeze(-1)
        quat_delta = so3_exp_to_quat_xyzw(domega_body)
        quat_next = quat_mul_xyzw(quat, quat_delta)
    else:
        p_next = p + t_delta_body
        quat_delta = so3_exp_to_quat_xyzw(domega_body)
        quat_next = quat_mul_xyzw(quat_delta, quat)

    quat_next = quat_next / (torch.linalg.norm(quat_next, dim=-1, keepdim=True) + 1e-12)
    return torch.cat([p_next, quat_next], dim=-1)