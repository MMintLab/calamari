import numpy as np


class TaskConfig:
    def __init__(self):
        # TODO : add variations of the language prompts
        self.language_prompts = {
            "wipe": ["wipe the dots up"],
            "sweep": ["sweep dirt to dustpan"],
            "push": ["push the red button"],
        }

        self.task_policy_configs = {
            "wipe": {
                "data_dir": "dataset/wipe",
                "contact_folder": "contact_front",
                "txt_cmd": self.language_prompts["wipe"][0],
                "tool_name": "sponge",
                "target_name": "diningTable",
                "train_idx": np.arange(0, 100),
                "test_idx": np.arange(100, 120),
                "heatmap_folder": "heatmap/",
            },
            "sweep": {
                "data_dir": "dataset/sweep_to_dustpan1",
                "contact_folder": "contact_front",
                "txt_cmd": self.language_prompts["sweep"][0],
                "tool_name": "broom",
                "target_name": "diningTable",
                "train_idx": np.arange(0, 100),
                "test_idx": np.arange(100, 120),
                "heatmap_folder": "heatmap/",
            },
            "push": {
                "data_dir": "dataset/push",
                "contact_folder": "contact_front",
                "txt_cmd": self.language_prompts["push"][0],
                "tool_name": "Panda_leftfinger_respondable",
                "target_name": "target_button_topPlate0",
                "train_idx": np.arange(0, 100),
                "test_idx": np.arange(100, 120),
                "heatmap_folder": "heatmap/",
            },
        }

        self.task_mpc_configs = None
        self.mpc_seed = None
