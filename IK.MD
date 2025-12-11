# 计算质心雅可比矩阵
        # 方法: J_com = d(com)/dq

# 使用 computeFrameJacobian 计算足端的雅可比矩阵,LOCAL
        J_full = cpin.computeFrameJacobian(
            self.cmodel, self.cdata, q, frame_id, pin.LOCAL
        )


