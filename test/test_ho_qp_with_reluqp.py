import torch
from wbc_ik_qp.ho_qp import Task, HoQp


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    n = 2000
    meq = 20
    mineq = 20
    # Build a small h matrix for z variables by creating a Task
    a = torch.randn((meq, n), device=dev, dtype=dtype)
    b = torch.randn((meq,), device=dev, dtype=dtype)
    d = torch.randn((mineq, n), device=dev, dtype=dtype)
    f = torch.randn((mineq,), device=dev, dtype=dtype)

    task = Task(a=a, b=b, d=d, f=f, device=dev, dtype=dtype)

    h = HoQp(task, device=dev, dtype=dtype)

    sol = h.getSolutions()
    print('Solution length:', sol.shape[0])
    print('First 5 entries:', sol[:5])


if __name__ == '__main__':
    main()
