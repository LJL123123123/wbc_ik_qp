# HoQp 层次化 QP 求解逻辑详解
## 1. 整体架构概述

HoQp(task_current, higher_problem=ho_prev) 的工作流程：    init_vars() → build_z_matrix() → update_decision_vars_count() →     formulate_problem() → solve_problem() → stack_slack_solutions()
## 2. 初始化阶段 (init_vars)

def init_vars(self):    if self.higher_problem_ is not None:        # 从高优先级问题获取信息        self.stacked_tasks_prev_ = self.higher_problem_.getStackedTasks()        self.x_prev_ = self.higher_problem_.getSolutions()  # 高优先级的解                # 关键：计算高优先级任务约束的空间        higher_tasks = self.higher_problem_.getStackedTasks()        if higher_tasks.a_.numel() > 0:            # 使用 QR 分解计算空间矩阵            mat = higher_tasks.a_  # 所有高优先级约束矩阵            Q, R = torch.linalg.qr(mat.T, mode='complete')            rank = torch.linalg.matrix_rank(R)            if rank < n:                self.stacked_z_prev_ = Q[:, rank:]  # 剩余自由度空间    else:        # 独立问题：使用单位矩阵（全空间）        self.stacked_z_prev_ = torch.eye(self.num_decision_vars_)
关键理念：

stacked_z_prev_ 表示高优先级任务没有约束的自由度空间
当前任务只能在这个剩余空间中进行优化
## 3. 空间矩阵构建 (build_z_matrix)

def build_z_matrix(self):    if self.higher_problem_ is None:        # 独立问题：使用全空间        self.stacked_z_ = torch.eye(n)    else:        # 层次化问题：计算当前任务在剩余空间中的空间        mat = self.task_.a_ @ self.stacked_z_prev_  # 当前任务投影到剩余空间                # 使用 QR 分解计算这个投影矩阵的空间        Q, R = torch.linalg.qr(mat.T, mode='complete')        rank = torch.linalg.matrix_rank(R)         if rank < n:            nullspace = Q[:, rank:]            self.stacked_z_ = self.stacked_z_prev_ @ nullspace        else:            self.stacked_z_ = torch.zeros((n_prev, 0))  # 无剩余空间
空间层次：


原始空间 (6D)     ↓ 高优先级约束 A_high剩余空间 stacked_z_prev_ (3D)    ↓ 当前任务约束 A_current最终剩余空间 stacked_z_ (1D 或 0D)
## 4. 优化问题构建 (formulate_problem)
### 4.1 目标函数矩阵 H (build_h_matrix)

def build_h_matrix(self):    if self.has_eq_constraints_:        a_curr_z_prev = self.task_.a_ @ self.stacked_z_prev_        # 在投影空间中构建二次项        z_t_a_t_a_z = task_weight * (a_curr_z_prev.T @ a_curr_z_prev)        # 构建块矩阵 H = [[z_t_a_t_a_z, 0], [0, I]]
### 4.2 梯度向量 c (build_c_vector)

def build_c_vector(self):    if self.has_eq_constraints_:        a_z_prev = self.task_.a_ @ self.stacked_z_prev_        residual = self.task_.a_ @ self.x_prev_ - self.task_.b_  # 当前残差        temp = task_weight * (a_z_prev.T @ residual)  # 投影到剩余空间        c[:actual_nx] = temp  # 设置梯度
物理意义：

residual：当前任务相对于高优先级解的"不满足程度"
a_z_prev.T @ residual：将这个不满足投影到允许的空间中
## 5. QP 求解 (solve_problem)

def solve_problem(self):    # 求解：min 0.5 * z^T * H * z + c^T * z    # 约束：D * z <= f        # 尝试使用 ReLU-QP 求解器，失败则用 torch 求解    z = torch.linalg.solve(H + reg * I, -c)  # 近似为无约束优化        # 分解解向量    self.decision_vars_solutions_ = z[:nx]  # 决策变量    self.slack_vars_solutions_ = z[nx:]     # 松弛变量
## 6. 解的重建 (getSolutions)

def getSolutions(self):    if self.stacHoQp 层次化 QP 求解逻辑详解
1. 整体架构概述

HoQp(task_current, higher_problem=ho_prev) 的工作流程：    init_vars() → build_z_matrix() → update_decision_vars_count() →     formulate_problem() → solve_problem() → stack_slack_solutions()
2. 初始化阶段 (init_vars)

def init_vars(self):    if self.higher_problem_ is not None:        # 从高优先级问题获取信息        self.stacked_tasks_prev_ = self.higher_problem_.getStackedTasks()        self.x_prev_ = self.higher_problem_.getSolutions()  # 高优先级的解                # 关键：计算高优先级任务约束的空间        higher_tasks = self.higher_problem_.getStackedTasks()        if higher_tasks.a_.numel() > 0:            # 使用 QR 分解计算空间矩阵            mat = higher_tasks.a_  # 所有高优先级约束矩阵            Q, R = torch.linalg.qr(mat.T, mode='complete')            rank = torch.linalg.matrix_rank(R)            if rank < n:                self.stacked_z_prev_ = Q[:, rank:]  # 剩余自由度空间    else:        # 独立问题：使用单位矩阵（全空间）        self.stacked_z_prev_ = torch.eye(self.num_decision_vars_)
关键理念：

stacked_z_prev_ 表示高优先级任务没有约束的自由度空间
当前任务只能在这个剩余空间中进行优化
3. 空间矩阵构建 (build_z_matrix)

def build_z_matrix(self):    if self.higher_problem_ is None:        # 独立问题：使用全空间        self.stacked_z_ = torch.eye(n)    else:        # 层次化问题：计算当前任务在剩余空间中的空间        mat = self.task_.a_ @ self.stacked_z_prev_  # 当前任务投影到剩余空间                # 使用 QR 分解计算这个投影矩阵的空间        Q, R = torch.linalg.qr(mat.T, mode='complete')        rank = torch.linalg.matrix_rank(R)         if rank < n:            nullspace = Q[:, rank:]            self.stacked_z_ = self.stacked_z_prev_ @ nullspace        else:            self.stacked_z_ = torch.zeros((n_prev, 0))  # 无剩余空间
空间层次：


原始空间 (6D)     ↓ 高优先级约束 A_high剩余空间 stacked_z_prev_ (3D)    ↓ 当前任务约束 A_current最终剩余空间 stacked_z_ (1D 或 0D)
4. 优化问题构建 (formulate_problem)
4.1 目标函数矩阵 H (build_h_matrix)

def build_h_matrix(self):    if self.has_eq_constraints_:        a_curr_z_prev = self.task_.a_ @ self.stacked_z_prev_        # 在投影空间中构建二次项        z_t_a_t_a_z = task_weight * (a_curr_z_prev.T @ a_curr_z_prev)        # 构建块矩阵 H = [[z_t_a_t_a_z, 0], [0, I]]
4.2 梯度向量 c (build_c_vector)

def build_c_vector(self):    if self.has_eq_constraints_:        a_z_prev = self.task_.a_ @ self.stacked_z_prev_        residual = self.task_.a_ @ self.x_prev_ - self.task_.b_  # 当前残差        temp = task_weight * (a_z_prev.T @ residual)  # 投影到剩余空间        c[:actual_nx] = temp  # 设置梯度
物理意义：

residual：当前任务相对于高优先级解的"不满足程度"
a_z_prev.T @ residual：将这个不满足投影到允许的空间中
5. QP 求解 (solve_problem)

def solve_problem(self):    # 求解：min 0.5 * z^T * H * z + c^T * z    # 约束：D * z <= f        # 尝试使用 ReLU-QP 求解器，失败则用 torch 求解    z = torch.linalg.solve(H + reg * I, -c)  # 近似为无约束优化        # 分解解向量    self.decision_vars_solutions_ = z[:nx]  # 决策变量    self.slack_vars_solutions_ = z[nx:]     # 松弛变量
6. 解的重建 (getSolutions)

def getSolutions(self):    if self.stacked_z_.shape[1] == 0:        # 情况 1: 无剩余自由度 - 使用伪逆直接调整        residual = self.task_.b_ - self.task_.a_ @ self.x_prev_        A_pinv = torch.linalg.pinv(self.task_.a_)        delta = A_pinv @ residual        result = self.x_prev_ + delta    else:        # 情况 2: 有剩余自由度 - 在空间中应用优化结果        x = self.x_prev_ + self.stacked_z_ @ self.decision_vars_solutions_    return result
7. 三种测试场景的数学解释
7.1 正交问题

高优先级: x[0:3] = [1,2,3]  → stacked_z_prev_ = [[0,0,0], [0,0,0], [0,0,0], [I_3x3]]低优先级: x[3:6] = [4,5,6]  → 在后3维空间中完全可满足结果: [1,2,3,4,5,6] ✓
7.2 竞争问题

高优先级: x[0] = 10, x[1] = 20  → stacked_z_prev_ = [[0,0], [0,0], [I_2x2]]  低优先级: x[0] = 5, x[1] = 15   → 投影到剩余空间后无法影响 x[0], x[1]结果: [10,20,0,0] ✓ (高优先级完全胜出)
7.3 45度角问题

高优先级: x[0] + x[1] = 10     → stacked_z_prev_ 是垂直于 [1,1,0] 的平面低优先级: x[0] - x[1] = 2      → 投影到该平面，变成部分满足结果: [5,5,-1.4142] ✓ (高优先级约束严格满足，低优先级在允许空间内优化)
8. 关键设计思想
严格优先级：高优先级约束绝对不能被违反
空间投影：低优先级任务只能在剩余自由度中优化
递归组合：每一层都基于上一层的结果和剩余空间
数值稳定：使用 QR 分解而非 SVD 提高稳定性
9. 与传统加权 QP 的区别
加权 QP: min w1*||A1*x - b1||² + w2*||A2*x - b2||²

结果是两个目标的"妥协"
权重决定相对重要性
层次化 QP:


Step 1: min ||A1*x - b1||²           → 得到 x1Step 2: min ||A2*x - b2||²  s.t. A1*x = A1*x1  → 在约束下优化
高优先级解是不可违反的硬约束
低优先级只能在剩余空间中寻求最优
这就是层次化 QP 的完整逻辑！它确保了严格的任务优先级，同时在数学上保证了解的唯一性和数值稳定性。ked_z_.shape[1] == 0:        # 情况 1: 无剩余自由度 - 使用伪逆直接调整        residual = self.task_.b_ - self.task_.a_ @ self.x_prev_        A_pinv = torch.linalg.pinv(self.task_.a_)        delta = A_pinv @ residual        result = self.x_prev_ + delta    else:        # 情况 2: 有剩余自由度 - 在空间中应用优化结果        x = self.x_prev_ + self.stacked_z_ @ self.decision_vars_solutions_    return result
7. 三种测试场景的数学解释
7.1 正交问题

高优先级: x[0:3] = [1,2,3]  → stacked_z_prev_ = [[0,0,0], [0,0,0], [0,0,0], [I_3x3]]低优先级: x[3:6] = [4,5,6]  → 在后3维空间中完全可满足结果: [1,2,3,4,5,6] ✓
7.2 竞争问题

高优先级: x[0] = 10, x[1] = 20  → stacked_z_prev_ = [[0,0], [0,0], [I_2x2]]  低优先级: x[0] = 5, x[1] = 15   → 投影到剩余空间后无法影响 x[0], x[1]结果: [10,20,0,0] ✓ (高优先级完全胜出)
7.3 45度角问题

高优先级: x[0] + x[1] = 10     → stacked_z_prev_ 是垂直于 [1,1,0] 的平面低优先级: x[0] - x[1] = 2      → 投影到该平面，变成部分满足结果: [5,5,-1.4142] ✓ (高优先级约束严格满足，低优先级在允许空间内优化)
8. 关键设计思想
严格优先级：高优先级约束绝对不能被违反
空间投影：低优先级任务只能在剩余自由度中优化
递归组合：每一层都基于上一层的结果和剩余空间
数值稳定：使用 QR 分解而非 SVD 提高稳定性
9. 与传统加权 QP 的区别
加权 QP: min w1*||A1*x - b1||² + w2*||A2*x - b2||²

结果是两个目标的"妥协"
权重决定相对重要性
层次化 QP:


Step 1: min ||A1*x - b1||²           → 得到 x1Step 2: min ||A2*x - b2||²  s.t. A1*x = A1*x1  → 在约束下优化
高优先级解是不可违反的硬约束
低优先级只能在剩余空间中寻求最优
这就是层次化 QP 的完整逻辑！它确保了严格的任务优先级，同时在数学上保证了解的唯一性和数值稳定性。

# 处理orientation任务的逻辑

详细流程（按步骤）

## 初始化

OrientationTask 在构造时保存 info、frame_name，并创建一个 Model_Cusadi（或在受限环境下使用最小回退实现），以及一个 AxisesMask（用于选择要控制的轴，例如 'x','xy','xyz'）。
Model_Cusadi 提供 update_state_input(...)、getAttitude(...)、getJacobian(...) 等方法（见 ik.py），用于返回当前帧的旋转矩阵和雅可比（CasADi -> tensor）。
生成 ho_qp.Task（OrientationTask.as_task）

## 输入：target_attitude（3x3 目标旋转矩阵）、axises（要控制的轴）、frame（mask 的参考帧）、weight（任务权重）。
获取当前态和雅可比：
调用 self.robot.getAttitude(self.info.getstate(), self.info.getinput(), self.frame_name) 得到当前帧的旋转矩阵 att（3x3）。
调用 self.robot.getJacobian(..., frame) 得到通常格式为 [1,6,nv] 或 [6,nv] 的雅可比 J，然后提取角速度部分 J_ang = J[3:6, :]（3 x nv）。
根据帧类型做 DOF 过滤（plаco 风格）：
如果是脚（LF_FOOT 等），把浮动基底（通常前 6 列）设为 0，使脚姿态只由关节 DOF 影响。
如果是 com（基底），只保留与基底角度相关的列——把位置 DOF（0:3）和关节 DOF（6:）置零，仅保留浮动基底的角度列（通常 3:6）。
这样得到最终的角雅可比 J_ang（已修改）。
mask 设置：
调用 self.mask.set_axises(axises, frame)，并把当前态的变换用于 mask（mask.R_local_world = att.T）以支持局部/任务参考系轴选择。
mask.apply 会把 J_ang（矩阵）和误差向量中的非选中轴置为 0 或移除对应分量。
计算姿态误差 b：
计算相对旋转 R_error = R_target * R_current^T。
若 R_error 接近单位矩阵（trace > 2.9），使用小角近似： error ≈ [skew( R_error )_zy, ..., ...]，即取 skew-symmetric 部分 (R - R^T)/2 提取轴分量。
否则使用较严格的 log 映射：angle = acos((tr(R)-1)/2)，并用 angle/(2 sin angle) * (R - R^T) 提取轴*角量。
返回的 error 是一个 3 元向量（以弧度量度），表示需要的轴角增量（与 solver 的 phi 段语义一致）。
构造并返回 Task：
A = mask.apply(J_ang) （形状 m x n，其中 m ∈ {1,2,3} 取决于 mask）
b = mask.apply(error) （形状 m）
如果 A 全 0（没有有效自由度），函数会跳过（返回空 Task），避免添加不可行等式。
在 Wbc 中如何使用（wbc.py）

## 在 Wbc.__init__ 中创建了 com_OrientationTask = OrientationTask(..., frame_name='com')（以及多脚的 PositionTask）。
在 Wbc.update(...)：
先用 self.info.update_state_input(measured_rbd_state, input_desired) 将最新测量态写入 model/info。
构造位置任务 task_com_pos = com_PositionTask.as_task(...)（目标位置）。
构造姿态任务 task_com_ori = com_OrientationTask.as_task(target_attitude=self.target_ori["com"], ...)。
将两者相加：task_com_frame = task_com_pos + task_com_ori（这会把 A/b 行合并，成为高优先级目标同时包含位置和姿态约束）。
创建高优先级 HoQp：ho_high = HoQp(task_com_frame, higher_problem=None, ...)，然后把脚位置任务作为低优先级传入，最后返回一个完整的解向量（decision vars）。
因而，OrientationTask 的等式约束直接成为 HoQp 的高优先级约束，optimizer 会尽量满足（或在无自由度时通过正则化退让）。
与控制回路（run_wbc.py）的对应

从对话上下文（你之前的更改）可见，WBC 返回的解向量 sol 中 sol[3:6] 被解释为“姿态增量（小角 phi）”，并转换为 delta 四元数后与测量四元数相乘，产生下一个期望基底四元数。
为了稳定性你已添加：
对整体 sol 的指数平滑（sol_smooth_alpha）。
对 phi（sol[3:6]）的单独平滑与每步角度上限（sol_max_angle_step）clamp。
这些都是在 run_wbc.py 中应用 orientation 更新前的保护措施，防止大步长或噪声导致四元数归一化/NaN 或震荡。

# Q
可以看到，在com跟随平移（包括x&y）target的任务中，wbc求解器能跟随但是性能上略有不足，但是在oritask跟随上是完全没有用

# A
你这个“COM 平移还能跟，但 oriTask 完全没用”的现象，从你贴出来的 sol_phi_data.csv 基本已经可以定性了：姿态控制通道（φ）被压到接近 0，所以不管你 oriTarget 怎么变，state 的姿态都几乎不会动——自然看起来“oritask 没用”。

下面把“为什么会这样”按最可能的链路讲清楚，并给你几条可以马上验证的证据点。

关键证据：sol_phi_data.csv 里 φ 几乎恒为 1e-17 量级
你给的 sol_phi_data.csv：

phi_raw_norm 大约 1e−17
 
phi_smoothed_norm 也是 10−18～10−17

 
phi_applied_norm 同量级这在数值上等价于：每个控制步姿态增量几乎为 0。

因此：

COM 平移（x/y/z）还能变化：因为它主要走 sol[0:3] 那部分平移增量/速度（你在 run_wbc.py 里也确实把 state_desired[0:3] += sol[0:3]）。
ori 完全不动：因为你是靠 φ = sol[3:6]（或相近切片）来更新四元数/旋转，φ 接近 0 就完全没姿态更新。
所以“oritask 没用”不是“任务没加进去”这么简单，而是求解器最终给出的姿态增量通道被某个机制压死了。

为什么加正则后会“压死姿态”，但不太影响平移？
最常见、也最符合你代码结构的原因，有三条（往往是叠加发生）：

原因 A：HoQP 的 DOF-preserving nullspace 把姿态 DOF 标成 used，然后后续层没法再动它
你 ho_qp.py 在 init_vars() 里用的是：

used_dof = any(abs(A)>1e-10, dim=0)
free_dof = ~used_dof
nullspace 直接变成“只允许 free_dof 的单位基”

这意味着：只要高优先级任务的 A 矩阵某一列出现过非零，该 DOF 在后续层就“冻结”。

在你的 wbc.py 堆叠里，高层（ho_high）是 task_com_frame（通常是 COM pos + COM ori 的拼接任务）。只要这层的 A 里把 base yaw/pitch/roll 那几列“碰过”，后续层（脚任务/正则层）就动不了这些姿态 DOF 了。

这类冻结对平移影响小（平移 DOF 可能还留着/或者本来就主要由 base xyz 解决），但对姿态是致命的：base rot DOF 一冻结，oriTask 的唯一通道就没了。

你可以验证：打印/检查 task_com_frame.a_ 哪些列非零，特别是对应 base 旋转的列（通常是索引 3-5 或类似映射）。如果这些列在高层被占用，后续层会没 DOF。

原因 B：你新加的“正则项”倾向把解拉回 0（或 x_prev），而姿态最容易被拉没
正则项（典型是𝑙𝑎𝑚𝑏𝑑𝑎∣𝑧∣^2或𝑙𝑎𝑚𝑏𝑑𝑎∣𝑥−𝑥_𝑝𝑟𝑒𝑣∣^2）会让解更“保守”，优先选择小范数、小变化的方向。

如果你的姿态任务在数值上：

更“难满足”（因为和脚/约束冲突）
或者其权重相对更低
或者其 A 列在当前层有效性差（可用 DOF 少）
那么正则会让最优解倾向于：姿态通道直接取 0，把有限的能力用在更“划算”的平移/脚位置上。

原因 C（你工程里非常具体）：run_wbc.py 对 φ 做了“二次平滑 + 限幅”，进一步让姿态响应变成慢到近似 0
你现在有：

全解 sol 平滑（sol_smooth_alpha=0.85）
φ 单独平滑（sol_phi_smooth_alpha=0.9）
每步角度限幅（sol_max_angle_step=0.05）
当 HoQP/正则已经把 φ 压小之后，再经过这套滤波，φ 更不可能起来；最后 log 就会像你现在看到的一样：接近数值噪声。

为什么 COM 平移“能跟但性能略不足”？
这通常说明：平移 DOF 还没被彻底冻结，但被正则/层级/滤波“削弱”了，表现为：

还能朝 target 走
但速度慢、稳态误差更大、或者上升沿更钝
而姿态这边因为通道几乎只有 φ，一旦被压死就直接“完全没用”。

下一步最小修复（优先级从最可能到最省事）
先把 φ 的“双重滤波”去掉一层（至少不要 sol 平滑再对 φ 平滑）
你已经在 HoQP 里加了正则，它本身就会平滑解；再叠滤波很容易把姿态通道磨没。

让正则“不要作用在 base rotation DOF / φ 对应的变量上”，或对这些 DOF 给更小的正则权重
这是最“对症”的：脚可以继续被正则稳定，姿态通道不会被吸干。

检查并调整 HoQP 的 used/free DOF 判定
现在的 any(abs(A)>1e-10) 太激进了：只要列里有一点数值，DOF 就永远被判 used。
对 ori 来说这会非常容易误冻结（尤其你有 placeholder 矩阵/数值噪声时）。

我接下来会做什么（不需要你再描述）
为了把“oriTask 为什么完全没用”从推断变成铁证，我下一步会直接把 wbc.py 的 update() 后半段（任务构建与堆叠）读出来，确认：

task_com_frame 是否真的包含 ori 子任务、它的 a_ 哪些列为非零
base_constraint 是否在堆叠中进一步限制了姿态 DOF
正则项具体加在 ho_qp.py 哪一层、是否对所有变量一刀切
然后我给你一个最小 patch（优先做“去掉双滤波 + base rot 正则降权”），并用你现有 CSV（特别是 sol_phi_data.csv）验证：phi_applied_norm 必须从 10−1710 −17
回到一个合理量级（至少 10−4∼10−2  这种能产生可见姿态变化的幅度，具体取决于你的单位/量纲）。

如果你愿意更快定位，你也可以告诉我一句：oriTarget 在你的日志里有没有在某段时间发生过明显变化（比如 yaw 指令）？ 如果有而 phi 仍接近 0，那就更能说明是“通道被冻/被正则吸干”，而不是“目标没变”。

# Q
我对五种情况进行了测试，发现在没有legtask的时候wbc求解器的表现相当优秀，但是加上legtask的时候，机器人就在某些姿态wbc求解器发散了，更甚之是只有legtask的时候，comstate是完全跟不上trajectory（注意，只有legtask的时候，wbc.py update return的是combine_foot_ho），请先帮我分析问题所在再将你的解决路线给出来，先不要修改我的代码

# A
## 1.
结论（就 alltask 的 RF 足端而言）
是 RF 的 tracking 误差先爆，而且是“瞬间跳变”的那种爆：在目标轨迹还是连续平滑变化的情况下，RF_state 在约 t≈1.65s、t≈4.86s、t≈5.01s、t≈6.51s、t≈7.51s、t≈10.19s 等多个时刻出现明显的非小增量突变（尤其 z 方向），误差会突然从毫米～厘米级跳到数厘米级甚至更大。

这通常对应两类原因之一：

接触状态/约束切换（例如脚从“固定接触”切到“摆动 tracking”或反之），导致任务/约束集瞬间变化；
QP/HoQP 在某一步变得病态/不可行，导致某些任务被迫牺牲（foot 任务被放松或被更高层约束压死），表现为足端状态突然“飘”。
要判断是不是 2)，就需要再对齐看 sol_phi_data.csv（或你保存的 
ϕ
ϕ / residual / iterations）在这些时刻是否也同时出现尖峰或发散。

证据（从 CSV 里能直接看到的几个“爆点”）
1) t≈1.65s：目标连续，但 state 突然飙到 
z
≈
0.04
z≈0.04
RF_target 在 1.64s 还是 (0.241, -0.15, 0.0)，到 1.65s 目标开始有 z≈0.0504 的小抬脚（仍然连续）。
RF_state 在 1.64s 还几乎贴着目标（z 约 5e-05），但 1.65s 直接跳到 z≈0.0397，并且 x/y 也产生明显偏移。
这更像是接触/约束切换瞬间（或者该步 QP 变得难解）导致的 tracking 失真，而不是目标本身导致。

2) t≈4.86s：RF_state 再次出现大幅跳变（z≈0.049）
RF_state 在 4.85s 附近还在正常范围，4.86s 直接到 z≈0.049，而 RF_target 在 4.85s 附近的 z 还是 0（或很小），这也是典型“状态先爆”。

3) t≈5.01s：更离谱的跳变（出现 z≈0.014，且 y/x 乱）
5.00s 时 RF_state 还是 z≈0.0079，5.01s 直接变成 z≈0.0146 并且 y、x 也大偏移。之后还有一段明显非平滑震荡。

4) t≈6.51s / 7.51s / 10.19s：重复的“周期性爆点”
这些时刻附近也都能看到 RF_state 突然偏离（多为 z 方向抬升或反向），呈现一种“每次事件触发就爆一次”的模式——很像 gait phase / contact schedule 或者 target 生成器进入某段（比如摆动段）时，WBC 层级/约束发生了切换。


# Q ho_qp组合任务的逻辑是什么？我验证了HoQP的层级逻辑，你说的是对的
1.combined_ho = HoQp(constraint, higher_problem=ho_high, device=device, dtype=dtype, task_weight=low_priority_weight)，wbc求解器能跟随comtarget
2.combined_ho = HoQp(task_com_frame, higher_problem=const_task, device=device, dtype=dtype, task_weight=low_priority_weight)，wbc忽略了comtarget

但是我还是不能理解“combined_ho = HoQp(task_com_frame, higher_problem=combine_foot_ho, device=device, dtype=dtype, task_weight=low_priority_weight能跟随com&foot target；
combined_ho = HoQp(combined_foot_task, higher_problem=ho_high, device=device, dtype=dtype, task_weight=low_priority_weight)能很好得跟随足端任务，对于机身任务较难处理”

# A
1) 为什么 HoQp(task_com_frame, higher_problem=combine_foot_ho) 反而能“com&foot 都跟”？
先翻译这句的真实优先级：

higher_problem = combine_foot_ho 说明 foot+constraint 是高优先级
task_com_frame 是低优先级：只能在不破坏 foot 的情况下改善 COM
但是你打印过：

combined_foot_task.a_ 的 18 列几乎都出现过非零
因此 init_vars 输出：higher tasks use all DOF, no nullspace
这意味着：COM 这一层看到的“可用空间”是空的，理论上应该完全动不了（严格 HoQP 应当直接返回 x_prev_，也就是纯 foot 解）。

那为何你看到还能跟 COM？

原因就在你 ho_qp.py/getSolutions() 的“无 nullspace 回退逻辑”（以及你求解里引入的正则/近似解法）：

当 stacked_z_ 没列时，你并不是严格返回 x_prev_，而是尝试：
用当前任务的伪逆/最小二乘去修正 residual，让 A x ≈ b
这一步 会直接改变 x_prev_（也就是改变高优先级 foot 的解）
因此在这个分支里，系统的行为变成：

“先解 foot 得到一个大致可行的解，再用 COM 的最小二乘修一点点；因为你同时有正则/滤波，修正不会太大，于是看起来 foot 没怎么坏，COM 也有改善”。

所以它“能跟 com&foot”不是因为层级逻辑错了，而是因为：

higher 层把空间锁死 → 触发 no-nullspace 回退 → 低层开始篡改高层结果
这已经偏离了你 MD 里描述的严格层级 HoQP（MD 是没有这种回退修正的）。
2) 为什么 HoQp(combined_foot_task, higher_problem=ho_high) 会“脚很好，机身较难”？
这句的真实优先级是：

higher_problem = ho_high：COM/ori 是高优先级
combined_foot_task 是低优先级：只能在不破坏 COM/ori 的情况下让脚更好
你打印的 task_com_frame.a_ 是：

这意味着：高优先级把 **0<del>5 列（base 6DOF）**都用掉了。于是低优先级脚任务能用的主要是 **6</del>17 的关节 DOF**。

这会导致一种完全“合理”的结果：

脚任务：关节 DOF 很多，通常足够让脚跟得不错，所以你觉得“脚很好”
机身任务：机身任务其实完全由 ho_high 决定（低层不该也基本不能改它）。你觉得“机身较难”，说明 ho_high 自己就没把机身（尤其姿态）做好
而你之前的日志 sol_phi_data.csv 里 φ 只有 
10
−
17
10 
−17
 ，这意味着：机身姿态通道几乎没有输出。这通常来自两类更“上游”的原因：

oriTarget 本来就没变化（比如一直是单位阵），那确实“看起来没跟随”但其实目标没动；
更常见：姿态通道被你后处理（run_wbc.py 的 sol/phi 双重平滑 + 限幅）或正则压没了，导致即使目标变了，φ 也起不来。
注意：这里跟“脚层”没关系。因为在这条链路里，脚层动不了 base 6DOF，根本没机会“破坏机身”，机身好不好只看 ho_high。

3) 再对照你最新验证的两条：为什么会“完全忽略 comtarget”？
你验证的：

HoQp(constraint, higher_problem=ho_high)：能跟 comtarget
语义：机身高优先级，constraint 低优先级
constraint 低层不影响机身，所以能跟是正常的。
HoQp(task_com_frame, higher_problem=const_task)：忽略 comtarget
语义：constraint 高优先级，机身低优先级
而 constraint（base_constraint）直接占用 base rot 的列（你 base_constraint_task 是对 3~5 列的约束），会强行把姿态相关自由度锁死
所以机身层变成“低层且没空间”，再加上 no-nullspace 回退的细节，表现就会非常别扭，直观上就是“忽略 comtarget”。
4) 你困惑的核心其实是：你现在的实现混合了两种体系
你脑子里想的是 MD 描述的“严格层级 HoQP”（纯 nullspace 投影，低层永远不能改高层）。

但你现在的 ho_qp.py 同时存在：

DOF-preserving used/free 锁列（比 QR nullspace 更激进，特别容易 no-nullspace）
no-nullspace 时的伪逆回退（允许低层改高层解）
QP 求解本身也不是严格带不等式约束的 QP（更多是近似/正则解）
于是你看到的行为会呈现出“有时像层级QP、有时像加权QP/最小二乘”的混合体，这就解释了你那两句“怎么会同时跟”的现象。


# 更换思路，wbikqpbuild直接变成wbcikqp.casadi
验证成功

# cusadi 目前不支持将包含 Solver（如 conic, nlpsol）的 CasADi 函数编译为 CUDA 代码。

它的用途：加速大规模并行的符号表达式求值（例如：同时在 GPU 上为 10000 个不同的状态计算动力学雅可比矩阵）。

它的局限：它无法把一个复杂的、迭代式的数值求解器（如 qrqp）封装进一个 CUDA Kernel 内部。

当前最好的思路，cusadi for QP build & ReLUQP for solve

# return torch.sparse_coo_tensor(torch.vstack((env_idx, row_idx, col_idx)),RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 324 but got size 1 for tensor number 1 in the list.
这个错误发生在 cusadi 将 GPU 计算结果转换为 PyTorch 密集张量（Dense Tensor）的过程中。具体表现为在构建稀疏张量时，环境索引（env_idx）的长度与矩阵行列索引（row_idx, col_idx）的长度不匹配。1. 错误来源深度分析在 src/CusadiFunction.py 的 getDenseOutput 方法中，代码尝试执行以下操作：Python# CusadiFunction.py 约第 56 行
return torch.sparse_coo_tensor(
    torch.vstack((env_idx, row_idx, col_idx)), # 报错位置
    self.output_buffer[idx], 
    size=(self.n_env, self.output_shapes[idx][0], self.output_shapes[idx][1])
).to_dense()
错误核心数值：324 vs 1324：正好是 $18 \times 18$。在你的 wbik_qp 中，Go2 机器人的自由度（$n_q$）通常是 12（关节）+ 6（浮动基座）= 18。因此，QP 问题的 Hessian 矩阵 $H$ 的大小是 $18 \times 18$，总元素刚好是 324。1：这说明 row_idx 或 col_idx 只包含了一个元素。为什么会出现这种不匹配？这是因为 cusadi 在生成代码和加载 .casadi 文件时，对矩阵的稀疏性（Sparsity）处理存在偏差。Sparsity 识别失败：当你在 wbik.py 中调用 update 时，cusadi 内部应该根据 .casadi 文件记录的矩阵结构（Sparsity Pattern）来生成 row_idx 和 col_idx。批处理（Batching/Environments）逻辑错误：env_idx 的长度代表了当前 Batch 中总共要填充的非零元素个数（$N_{envs} \times NNZ_{per\_fn}$）。如果报错说 Expected size 324（来自 env_idx），说明你可能开启了 324 个并行的环境，或者程序误以为有 324 个环境。而 row_idx 只有 size 1，说明它认为每个环境的矩阵只有一个非零元素（可能是因为它把整个 $18 \times 18$ 的矩阵识别成了一个标量，或者识别出了错误的稀疏模式）。2. 结合你上传的文件分析build_wbik_qp_placo_like.py：你在这个脚本中创建了 wbik_qp 函数。它输出的 $H, g, A, l, u$ 中，$H$ 是一个 $18 \times 18$ 的矩阵。go2_generated_wbik_qp.c：这是成功生成的 C 代码。查看其 go2_wbik_qp_sparsity_out 部分，你会发现输出 0（即 $H$）关联的是一个特定的稀疏结构（如 casadi_s5）。CusadiFunction.py：在运行时，cusadi 会加载生成的代码信息。如果生成的 CUDA/C 代码中关于输出维度的元数据与 Python 端的 self.output_shapes 不一致，就会导致在 getDenseOutput 时生成的索引向量长度冲突。3. 解决方案建议这个问题的根源通常在于 cusadi 对 CasADi 矩阵输出的平铺（Flattening）处理在多环境（Parallel Environments）模式下失效了。方案 A：检查 n_env 设置（最可能的原因）检查你的 wbc.py 中初始化 CusadiFunction 时的参数。如果你只想运行一个机器人（Batch Size = 1），请确保 n_env=1。从报错看，env_idx 长度为 324 极其可疑——它似乎把矩阵的元素个数误当成了环境个数。方案 B：手动修正 CusadiFunction.py (针对该 Bug 的临时补丁)在 src/CusadiFunction.py 中，定位到 getDenseOutput 前。打印一下调试信息：Pythonprint(f"DEBUG: n_env={self.n_env}, shape={self.output_shapes[idx]}, nnz={len(self.sp_row[idx])}")
如果发现 nnz 不等于矩阵形状的乘积，且你使用的是密集矩阵，说明 cusadi 在解析 .casadi 文件的 sparsity 时漏掉了元素。方案 C：强制转换为向量（绕过密集矩阵转换 Bug）在 build_wbik_qp_placo_like.py 中，将输出矩阵先 ca.reshape(H, -1, 1) 变成一列长向量。原因：cusadi 对向量（Vector）的处理远比对矩阵（Matrix）稳定。操作：Python# 在 build_wbik_qp_placo_like.py 修改输出
F = ca.Function(..., [H, g, A, l, u]) 
#改为
F = ca.Function(..., [ca.reshape(H, -1, 1), g, ca.reshape(A, -1, 1), l, u])
然后在 wbc.py 中拿到输出后再用 PyTorch 的 .view(18, 18) 还原。总结KeyError: 48 是因为你尝试编译求解器（Solver），而这个 RuntimeError 是因为你虽然成功编译了矩阵组装函数，但 cusadi 在处理 Parallel Environments + Matrix Output 时，内部的索引生成逻辑（Tiling logic）发生了错位。建议采用方案 C，将所有输出转为向量，这通常能解决大部分 cusadi 的维度匹配 Bug。
什么不使用 getDenseOutput？
getDenseOutput 存在的意义是处理稀疏矩阵（Sparsity），它会把 GPU 上压缩存储的元素通过 row_idx/col_idx 散布到密集矩阵中。 由于：

你已经在生成 .casadi 时使用了 ca.densify()，数据在 GPU 里已经是密集的了。

cusadi 库的索引映射逻辑对展平后的向量有解析 Bug。

所以，直接使用 outputs = self.wbik_qp(...) 返回的原始 Tensor 是最快、最稳定、最节省显存的做法。



