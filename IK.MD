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

# Q
现在wbik在机身角度接近+-pi/2的时候求解器发散了，请帮我找出原因

# A
从你这个 debug.zip 里其实已经能看出“发散”的关键特征，而且它**刚好发生在 yaw 接近 -1.305rad（≈ -75°，逼近 -π/2）**的时候：

step_norms_data.csv 末段：dp_norm（你把 sol[:3] 当作 base 平移增量）从 0.09 一路指数级涨到 27m。

com_state_data.csv 同一段：com_x, com_y 直接跑到 几十米，而目标 com_target_data.csv 一直在 0.60m 左右。

这不是“QP 本身突然坏了”，而是一个非常典型的外环积分（integrate）坐标系用错导致的正反馈：误差越大 → 你更新越把位姿推向更错的方向 → 下一步误差更大 → dp 更大 → 爆炸。

最可能的根因：你把 dp 当成 world 增量加了，但它其实是 base/local 坐标系的增量

在 Pinocchio / Placo 的 free-flyer 里，nv=6+nj 的前 6 维是 SE(3) 的切空间增量，默认是右扰动（right-trivialized），也就是**body frame（本体系）**的 twist：

x[:3]：本体系线位移/线速度增量（不是 world）

x[3:6]：本体系角增量

Placo 用的是 pinocchio::integrate(model, q, dq)，它内部做的是（右乘）：

𝑇𝑘+1=𝑇𝑘⋅exp(Δ𝜉),Δ𝜉=[Δ𝑝𝑏𝑜𝑑𝑦,Δ𝜔𝑏𝑜𝑑𝑦]Tk+1=Tk⋅exp(Δξ),Δξ=[Δpbody,Δωbody]

所以平移更新必须是：

𝑝𝑤𝑜𝑟𝑙𝑑←𝑝𝑤𝑜𝑟𝑙𝑑+𝑅𝑤𝑜𝑟𝑙𝑑⋅(𝑉(Δ𝜔)Δ𝑝𝑏𝑜𝑑𝑦)pworld←pworld+Rworld⋅(V(Δω)Δpbody)而不是 p_world += dp_body。

当 yaw 变大（接近 ±π/2）时，R_world 把 body-x/body-y 旋到世界坐标，你如果没乘这个 R，修正方向会越来越“侧着来”，于是就出现你 log 里那种“越修越偏、dp 越来越大”的爆炸。

你 debug 里恰好是 yaw≈-1.305rad 时开始指数爆炸，这和“平移增量坐标系用错”高度吻合。
