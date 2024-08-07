import torch 

class NFoldInferenceModel(torch.nn.Module):
    def __init__(self, models):
        super(NFoldInferenceModel, self).__init__()
        modules = dict()
        for idx, m in enumerate(models):
            modules[f'model_{idx+1}'] = m
        self.n_models = len(models)
        self.__dict__['_modules'] = modules

    def forward(self, x):
        res = 0
        for model_id in range(1, self.n_models+1):
            res += getattr(self, f'model_{model_id}')(x)
        return res / self.n_models