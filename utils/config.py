class Config:

    # default setting
    def __init__(self):
        
        # self.parser = argparse.ArgumentParser()

        self.norm = None
        self.exploration_policy = None

        self.warmup_steps = 0
        self.batch_size =  32
        self.buffer_size = 1e6

        self.tau = 0.01
        self.gamma = 0.99

        # if using OU noise for exploration
        self.ou_theta = 0.15
        self.ou_mu = 0.0
        self.ou_sigma = 0.2

    # add custom setting
    def merge_config(self, custom_config):

        for key in custom_config.keys():
            setattr(self, key, custom_config[key])

