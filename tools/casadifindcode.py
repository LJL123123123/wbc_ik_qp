import casadi
for name in dir(casadi):
    if name.startswith('OP_'):
        if getattr(casadi, name) == 48: # 将 48 替换为你遇到的 KeyError 数字
            print(f"Opcode 48 is {name}")