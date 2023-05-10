import numpy as np

class TaskConfig():
    def __init__(self):
        # TODO : add variations of the language prompts
        self.language_prompts = {"wipe": ["Use the sponge to clean up the dirt.",
                                          "Use the eraser to clean up the dirt."],
                                    "sweep": ["Use the broom to brush the dirt into the dustpan"],
                                    "scoop": ["Scoop up the block and lift it with the spatula"],
                                    "press": ["Press the maroon button"]}


        self.task_policy_configs = {
        #     "wipe": {
        #     # "data_dir": "dataset/heuristics_0228",
        #     "data_dir": "dataset/wipe",
        #     "contact_folder": 'contact_front',
        #     "txt_cmd": self.language_prompts["wipe"][0],
        #     "tool_name": 'sponge',
        #     'target_name':'diningTable',
        #     "train_idx": np.arange(0,100),
        #     # "train_idx": np.concatenate([np.arange(0,45), np.arange(50,105)]),
        #     "test_idx": np.arange(100, 110),
        #     "heatmap_folder": "heatmap_huy_center_/"
        # },
            # "sweep": {
            #     "data_dir": "dataset/sweep",
            #     "contact_folder": 'contact_front',
            #     "txt_cmd": self.language_prompts["sweep"][0],
            #     "tool_name": 'broom',
            #     'target_name': 'diningTable',
            #     "train_idx": np.arange(0,3),
            #     # "train_idx": np.concatenate([np.arange(0,45), np.arange(50,105)]),
            #     "test_idx": np.arange(100, 110),
            #     "heatmap_folder": "heatmap_huy_center_/"
            # },
            "push": {
                "data_dir": "dataset/push_0",
                "contact_folder": 'contact_front',
                "txt_cmd": self.language_prompts["press"][0],
                " tool_name": 'Panda_leftfinger_respondable',
                'target_name': 'target_button_topPlate0',
                "train_idx": np.arange(0, 100),
                # "train_idx": np.concatenate([np.arange(0,95), np.arange(95,105)]),
                "test_idx": np.arange(100, 110),
                "heatmap_folder": "heatmap_huy_center_/"
            },
            # "scoop": {"data_dir": "dataset/scoop_spatula_",
            #           "contact_folder": 'contact_front',
            #           "txt_cmd": self.language_prompts["scoop"][0],
            #           "tool_name": 'spatula',
            #           'target_name': 'diningTable',
            #           "train_idx": np.concatenate([np.arange(0,95), np.arange(100,300)]),
            #           "test_idx": np.arange(95, 100)}
            }

        self.task_mpc_configs = None
        self.mpc_seed = None