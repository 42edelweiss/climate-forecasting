from statsmodels.tsa.holtwinters import ExponentialSmoothing

import torch

class HwesPredictor(torch.nn.Module):
    def forward(self,x):
        last_values=[]
        for r in x.tolist():
            model = ExponentialSmoothing(r, seasonal=None, trend='add', initialization_method="estimated")
            fit_model = model.fit()
            pred = fit_model.forecast(1)
            last_values.append([pred[0]])
        return torch.tensor(data=last_values)