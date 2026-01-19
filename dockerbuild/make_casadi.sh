# #all foot
# python3 run_codegen.py --fn=go1_all_feet_attitude
# python3 run_codegen.py --fn=go1_all_feet_kinematics

# #com
python3 run_codegen.py --fn=go1_com_attitude
python3 run_codegen.py --fn=go1_com_position
python3 run_codegen.py --fn=go1_com_jacobian
python3 run_codegen.py --fn=go1_com_velocity

# #single foot
python3 run_codegen.py --fn=go1_FL_foot_attitude
python3 run_codegen.py --fn=go1_FL_foot_position
python3 run_codegen.py --fn=go1_FL_foot_jacobian
python3 run_codegen.py --fn=go1_FL_foot_velocity

python3 run_codegen.py --fn=go1_FR_foot_attitude
python3 run_codegen.py --fn=go1_FR_foot_position
python3 run_codegen.py --fn=go1_FR_foot_jacobian
python3 run_codegen.py --fn=go1_FR_foot_velocity

python3 run_codegen.py --fn=go1_RL_foot_attitude
python3 run_codegen.py --fn=go1_RL_foot_position
python3 run_codegen.py --fn=go1_RL_foot_jacobian
python3 run_codegen.py --fn=go1_RL_foot_velocity

python3 run_codegen.py --fn=go1_RR_foot_attitude
python3 run_codegen.py --fn=go1_RR_foot_position
python3 run_codegen.py --fn=go1_RR_foot_jacobian
python3 run_codegen.py --fn=go1_RR_foot_velocity

python3 run_codegen.py --fn=go1_com_jacobian_world
python3 run_codegen.py --fn=go1_FL_foot_jacobian_world
python3 run_codegen.py --fn=go1_FR_foot_jacobian_world
python3 run_codegen.py --fn=go1_RL_foot_jacobian_world
python3 run_codegen.py --fn=go1_RR_foot_jacobian_world