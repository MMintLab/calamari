import numpy as np

class TaskConfig():
    def __init__(self):
        # TODO : add variations of the language prompts
        self.language_prompts = {"wipe": [
                                        "wipe the dots up",
                                        "wipe the dirt up",
                                          "use the sponge to clean up the dirt",
                                          "Use the eraser to clean up the dirt."],
                                    "sweep": [
                                              "sweep dirt to dustpan",
                                              "Use the broom to brush the dirt into the dustpan.",],
                                    "scoop": ["scoop up the cube and lift it with the spatula"],
                                    "press": ["push the red button"],
                                    "draw": ["draw the letter C"]} # should be push!!!


        self.task_policy_configs = {
        #     "draw": {
        #     # "data_dir": "dataset/heuristics_0228",
        #     "data_dir": "dataset/drawing",
        #     "contact_folder": 'contact_front',
        #     "txt_cmd": self.language_prompts["draw"][0],
        #     "train_idx": np.arange(0, 10),
        #     # "train_idx": np.concatenate([np.arange(0,45), np.arange(50,105)]),
        #     "test_idx": np.arange(100, 110),
        #     "heatmap_folder": "heatmap_huy_mask/"
        # },
        #     "wipe": {
        #     # "data_dir": "dataset/heuristics_0228",
        #     "data_dir": "dataset/wipe_0603_2",
        #     "contact_folder": 'contact_front',
        #     "txt_cmd": self.language_prompts["wipe"][0],
        #     "tool_name": 'sponge',
        #     'target_name':'diningTable',
        #     "train_idx": np.arange(0, 10),
        #     # "train_idx": np.concatenate([np.arange(0,45), np.arange(50,105)]),
        #     "test_idx": np.arange(10, 20),
        #     "heatmap_folder": "heatmap_huy_mask_by_sentence/"
        # },
            # "sweep": {
            #     "data_dir": "dataset/sweep_to_dustpan1/episodes",
            #     "contact_folder": 'contact_front',
            #     "txt_cmd": self.language_prompts["sweep"][0],
            #     "tool_name": 'broom',
            #     'target_name': 'diningTable',
            #     "train_idx": np.arange(0,100),
            #     # "train_idx": np.concatenate([np.arange(0,45), np.arange(50,105)]),
            #     "test_idx": np.arange(100, 120),
            #     "heatmap_folder": "heatmap_huy_mask_by_sentence/"
            #     # "heatmap_folder": "heatmap_huy_mask_filter/"
            # },
            "push": {
                "data_dir": "dataset/push_0515",
                "contact_folder": 'contact_front',
                "txt_cmd": self.language_prompts["press"][0],
                "tool_name": 'Panda_leftfinger_respondable',
                'target_name': 'target_button_topPlate0',
                "train_idx": np.arange(0, 10),
                # "train_idx": np.concatenate([np.arange(0,95), np.arange(95,105)]),
                "test_idx": np.arange(110, 120), #
                "heatmap_folder": "heatmap_huy_mask_by_sentence/",
                # "heatmap_folder": "heatmap_huy_mask/"
                # "heatmap_folder": "heatmap_huy_center_/"
            },
            # "scoop": {"data_dir": "dataset/scoop_0518",
            #           "contact_folder": 'contact_front',
            #           "txt_cmd": self.language_prompts["scoop"][0],
            #           "tool_name": 'spatula',
            #           'target_name': 'diningTable',
            #             "train_idx": np.arange(0, 100),
            #         #   "train_idx": np.concatenate([np.arange(0,95), np.arange(100,300)]),
            #           "test_idx": np.arange(100, 120),
            #         "heatmap_folder": "heatmap_huy_mask/"}
            }

        self.task_mpc_configs = None
        self.mpc_seed = None