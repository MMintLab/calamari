import numpy as np

class TaskConfig():
    def __init__(self):
        # TODO : add variations of the language prompts
        self.language_prompts = {"wipe": ["Use the sponge to clean up the dirt.",
                                          "Use the eraser to clean up the dirt."],
                                    "sweep": ["Use the broom to brush the dirt into the dustpan"],
                                    "scoop": ["Scoop up the block and lift it with the spatula"]}


        self.task_policy_configs = {
        #     "wipe": {
        #     "data_dir": "dataset/heuristics_0228",
        #     "contact_folder": 'contact_front',
        #     "txt_cmd": self.language_prompts["wipe"][0],
        #     "tool_name": 'sponge',
        #     'target_name':'diningTable',
        #     "train_idx": np.concatenate([np.arange(0,45), np.arange(50,200)]),
        #     "test_idx": np.arange(45, 50),
        # },
            "sweep": {
                "data_dir": "dataset/sweep_to_dustpan",
                "contact_folder": 'contact_front',
                "txt_cmd": self.language_prompts["sweep"][0],
                "tool_name": 'broom',
                'target_name': 'diningTable',
                "train_idx": np.arange(0, 45),
                "test_idx": np.arange(45, 50),
            },
            "scoop": {"data_dir": "dataset/scoop_spatula_",
                      "contact_folder": 'contact_front',
                      "txt_cmd": self.language_prompts["scoop"][0],
                      "tool_name": 'spatula',
                      'target_name': 'diningTable',
                      "train_idx": np.arange(0, 45),
                      "test_idx": np.arange(45, 50)}
            }

        self.task_mpc_configs = None
        self.mpc_seed = None