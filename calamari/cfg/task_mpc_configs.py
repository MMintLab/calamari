from calamari.cfg.task_policy_configs_v2 import TaskConfig
from l4c_rlbench.rlbench.tasks.desk_wipe_demo import WipeDesk, WipeDeskWb,WipeDeskHd, WipeDeskHd2
from l4c_rlbench.rlbench.tasks.sweep_to_dustpan_demo import SweepToDustpan, SweepToDustpan1, SweepToDustpanRod, SweepToDustpan2, SweepToDustpanShort,SweepToDustpan1Ver, SweepToDustpanVer, SweepToDustpan1Hor,SweepToDustpan1Ver1, SweepToDustpan1Hor1
from l4c_rlbench.rlbench.tasks.push_button_demo import PushButtons
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, JointVelocity

class TaskMPCConfig(TaskConfig):
    def __init__(self):
        super().__init__()
        self.mpc_seed = 100
        self.task_mpc_configs = {
            'wipe' : {
                'txt_cmd': self.language_prompts['wipe'],
                'tasks': [WipeDesk, WipeDeskWb, WipeDeskHd, WipeDeskHd2],  # Call by the index
                'tool_name': self.task_policy_configs['wipe']['tool_name'],
                'target_name' : self.task_policy_configs['wipe']['target_name'],
                'grasp_target_name': self.task_policy_configs['wipe']['grasp_target_name'],
                'demo_controller': EndEffectorPoseViaPlanning(),
            },
            'sweep': {
                'txt_cmd': self.language_prompts['sweep'],
                'tasks': [SweepToDustpan1, SweepToDustpan1Ver, SweepToDustpan1Hor, SweepToDustpan1Ver1, SweepToDustpan1Hor1, SweepToDustpan2, SweepToDustpanShort, SweepToDustpanRod, SweepToDustpan1Ver,SweepToDustpan1Hor],  # Call by the index
                'tool_name': self.task_policy_configs['sweep']['tool_name'],
                'target_name': self.task_policy_configs['sweep']['target_name'],
                'grasp_target_name': self.task_policy_configs['sweep']['grasp_target_name'],
                'demo_controller': JointVelocity(),
            },

            'push': {
                'txt_cmd': self.task_policy_configs['push']['txt_cmd'],
                'tasks': [PushButtons],  # Call by the index
                'tool_name': self.task_policy_configs['push']['tool_name'],
                'target_name': self.task_policy_configs['push']['target_name'],
                'grasp_target_name': None,
                'demo_controller': JointVelocity(),
            },
        }