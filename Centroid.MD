# Q1
直接使用self.info.getstate()传入self.robot.getPosition()不行，要使用state = torch.as_tensor(self.info.getstate(), device=self.device, dtype=self.dtype)，再将state传入self.robot.getPosition()才行，从输出看
self.info.getstate(): tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.], dtype=torch.float64)
state after info: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.], device='cuda:0', dtype=torch.float64)
二者的区别是有没有device='cuda:0'，请告诉我这是的话i什么导致的？

你看到的差别就是设备不同：self.info.getstate() 返回的是在 CPU 上的 tensor（所以 print 不显示 device 字段），而 state = torch.as_tensor(self.info.getstate(), device=self.device, dtype=...) 会把 tensor 放到 GPU（device='cuda:0'），从而与 self.robot/模型期望的设备对齐；因为设备不匹配，直接传 CPU tensor 给模型的方法会走不同的代码路径或触发后端错误，所以必须先把它移到正确的 device。

# solution
将Centroidal.py的矩阵/向量储存&输出变成GPU tensor 。
测试position_task.py中的pos，att, J, A_masked, b_masked都为device='cuda:0'。

 # Q2
 Traceback (most recent call last):
File "/home/ReLUQP-py/wbc_ik_qp/wbc.py", line 307, in <module>
sol = w.update(state_desired, input_desired, measured, mode=0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/ReLUQP-py/wbc_ik_qp/wbc.py", line 288, in update
ho3 = HoQp(task_LF_pos, higher_problem=ho_high, device=dev, dtype=dtype)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/ReLUQP-py/wbc_ik_qp/ho_qp.py", line 106, in init
self.formulate_problem()
File "/home/ReLUQP-py/wbc_ik_qp/ho_qp.py", line 160, in formulate_problem
self.build_h_matrix()
File "/home/ReLUQP-py/wbc_ik_qp/ho_qp.py", line 173, in build_h_matrix
a_curr_z_prev = self.task_.a_ @ self.stacked_z_prev_
~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x18 and 19x19)
root@zenbot-ljl-System-Product-Name:/home/ReLUQP-py#

Centroidal的内容和ik的内容默认是对的,这里应该是18维才对,请帮我纠正
HoQP维度不对

