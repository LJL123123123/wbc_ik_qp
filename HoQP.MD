# HoQp 层次化 QP 求解逻辑详解
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

初始化

OrientationTask 在构造时保存 info、frame_name，并创建一个 Model_Cusadi（或在受限环境下使用最小回退实现），以及一个 AxisesMask（用于选择要控制的轴，例如 'x','xy','xyz'）。
Model_Cusadi 提供 update_state_input(...)、getAttitude(...)、getJacobian(...) 等方法（见 ik.py），用于返回当前帧的旋转矩阵和雅可比（CasADi -> tensor）。
生成 ho_qp.Task（OrientationTask.as_task）

输入：target_attitude（3x3 目标旋转矩阵）、axises（要控制的轴）、frame（mask 的参考帧）、weight（任务权重）。
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

在 Wbc.__init__ 中创建了 com_OrientationTask = OrientationTask(..., frame_name='com')（以及多脚的 PositionTask）。
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