# 计算质心雅可比矩阵
        # 方法: J_com = d(com)/dq

# 使用 computeFrameJacobian 计算足端的雅可比矩阵,LOCAL
        J_full = cpin.computeFrameJacobian(
            self.cmodel, self.cdata, q, frame_id, pin.LOCAL
        )

# Q
ori_casadi = ori_casadi_world(x0)
print("Casadi Orientation:\n", ori_casadi)
衡为单位向量

# A
在casadi构建attitude库时，需要注意机身的link name，go1的urdf是以trunk构建的float Joint，base与trunk并不是一起的
现在已经改成了输入link name
