import torch
import torch

def quat_to_rotmat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (xyzw) to rotation matrix.
    q: shape (4,) or (B,4), format [x,y,z,w]
    returns: (3,3) or (B,3,3)
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)  # (1,4)
        squeeze = True
    else:
        squeeze = False

    # normalize
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-12)

    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    # Rotation matrix (right-handed, passive, mapping body->world)
    R = torch.stack([
        ww + xx - yy - zz, 2*(xy - zw),       2*(xz + yw),
        2*(xy + zw),       ww - xx + yy - zz, 2*(yz - xw),
        2*(xz - yw),       2*(yz + xw),       ww - xx - yy + zz
    ], dim=-1).reshape(-1, 3, 3)

    return R[0] if squeeze else R


def rotmat_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (xyzw).
    R: (3,3) or (B,3,3)
    returns: (4,) or (B,4) in [x,y,z,w]
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # From https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    m00 = R[:, 0, 0]; m01 = R[:, 0, 1]; m02 = R[:, 0, 2]
    m10 = R[:, 1, 0]; m11 = R[:, 1, 1]; m12 = R[:, 1, 2]
    m20 = R[:, 2, 0]; m21 = R[:, 2, 1]; m22 = R[:, 2, 2]

    tr = m00 + m11 + m22
    q = torch.zeros((R.shape[0], 4), dtype=R.dtype, device=R.device)

    # Compute w first for numerical stability
    w = torch.sqrt(torch.clamp(tr + 1.0, min=0.0)) * 0.5
    q[:, 3] = w

    # Avoid division by zero
    denom = 4.0 * w + 1e-12

    q[:, 0] = (m21 - m12) / denom
    q[:, 1] = (m02 - m20) / denom
    q[:, 2] = (m10 - m01) / denom

    # Normalize
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-12)
    return q[0] if squeeze else q

def skew(w: torch.Tensor) -> torch.Tensor:
    wx, wy, wz = w
    return torch.tensor([[0, -wz, wy],
                         [wz, 0, -wx],
                         [-wy, wx, 0]], dtype=w.dtype, device=w.device)

def so3_exp(w: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    if theta < 1e-8:
        W = skew(w)
        return I + W  # 1st order
    W = skew(w/theta)
    return I + torch.sin(theta)*W + (1-torch.cos(theta))*(W@W)

def so3_left_jacobian(w: torch.Tensor) -> torch.Tensor:
    # V(w) in SE3 exp: t = V(w) * v
    theta = torch.linalg.norm(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    W = skew(w)
    if theta < 1e-8:
        # series: I + 1/2 W + 1/6 W^2
        return I + 0.5*W + (1.0/6.0)*(W@W)
    theta2 = theta*theta
    A = (1 - torch.cos(theta)) / theta2
    B = (theta - torch.sin(theta)) / (theta2*theta)
    return I + A*W + B*(W@W)

def quat_mul_xyzw(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    # q,r: [x,y,z,w]
    x1,y1,z1,w1 = q
    x2,y2,z2,w2 = r
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def so3_exp_to_quat_xyzw(w: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(w)
    if theta < 1e-8:
        return torch.tensor([0,0,0,1], dtype=w.dtype, device=w.device)
    axis = w/theta
    half = 0.5*theta
    s = torch.sin(half)
    return torch.cat([axis*s, torch.cos(half).unsqueeze(0)])

def integrate_freeflyer_quat_xyzw(q_fb: torch.Tensor,
                                 dp_body: torch.Tensor,
                                 domega_body: torch.Tensor,
                                 right_multiply: bool = True) -> torch.Tensor:
    """
    q_fb: [px,py,pz, qx,qy,qz,qw] (world position + xyzw quaternion)
    dp_body,domega_body: freeflyer increment in BODY frame (Pinocchio/Placo convention)
    right_multiply=True corresponds to T_next = T * Exp(dxi)
    """
    p = q_fb[0:3]
    quat = q_fb[3:7]  # xyzw

    # current rotation
    # (you likely already have a quat->R; omitted here for brevity)
    R = quat_to_rotmat_xyzw(quat)  # implement your own

    R_delta = so3_exp(domega_body)
    V = so3_left_jacobian(domega_body)
    t_delta_body = V @ dp_body

    if right_multiply:
        # T_next = T * Exp(dxi)
        p_next = p + R @ t_delta_body
        quat_delta = so3_exp_to_quat_xyzw(domega_body)
        quat_next = quat_mul_xyzw(quat, quat_delta)
    else:
        # T_next = Exp(dxi) * T  (一般你不需要这个)
        p_next = p + t_delta_body
        quat_delta = so3_exp_to_quat_xyzw(domega_body)
        quat_next = quat_mul_xyzw(quat_delta, quat)

    quat_next = quat_next / torch.linalg.norm(quat_next)

    return torch.cat([p_next, quat_next])