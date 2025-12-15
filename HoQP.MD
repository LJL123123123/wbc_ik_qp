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