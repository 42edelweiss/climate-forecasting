import copy
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import get_time_series_datasets
from dummy_model import DummyPredictor
from fcnnn_model import FCNN
from hwes_model import HwesPredictor
from interpolation_model import InterpolationPredictor

random.seed(1)
torch.manual_seed(1)

features = 256
ts_len = 3_000

x_train, x_val, x_test, y_train, y_val, y_test = get_time_series_datasets(features, ts_len)

# CORRECTION ICI: n_inp → n_in
net = FCNN(n_in=features, l_1=64, l_2=32, n_out=1)
net.train()

dummy_predictor = DummyPredictor()
interpolation_predictor = InterpolationPredictor()
hwes_predictor = HwesPredictor()

optimizer = torch.optim.Adam(params=net.parameters())
loss_func = torch.nn.MSELoss()

best_model = None
min_val_loss = 1_000_000
training_loss = []
validation_loss = []

for t in range(10_000):
    prediction = net(x_train)
    loss = loss_func(prediction, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    val_prediction = net(x_val)
    val_loss = loss_func(val_prediction, y_val)
    
    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    
    if val_loss.item() < min_val_loss:
        best_model = copy.deepcopy(net)
        min_val_loss = val_loss.item()
    
    if t % 1000 == 0:
        print(f'epoch {t}: train - {round(loss.item(), 4)}, val: - {round(val_loss.item(), 4)}')

net.eval()

print('Testing')
print(f'FCNN Loss: {loss_func(best_model(x_test), y_test).item()}')
print(f'Dummy Loss: {loss_func(dummy_predictor(x_test), y_test).item()}')
print(f'Linear Interpolation Loss: {loss_func(interpolation_predictor(x_test), y_test).item()}')
print(f'HWES Loss: {loss_func(hwes_predictor(x_test), y_test).item()}')

plt.title("Training progress")
plt.yscale("log")
plt.plot(training_loss, label='training loss')
plt.plot(validation_loss, label='validation loss')
plt.legend()
plt.savefig('training_progress.png')  # Sauvegarde au lieu de show()
plt.close()

plt.title("FCNN on Train Dataset")
plt.plot(y_train, label='actual')
plt.plot(best_model(x_train).detach().numpy(), label='predicted')  # detach() pour éviter warnings
plt.legend()
plt.savefig('fcnn_train.png')
plt.close()

plt.title('Test')
plt.plot(y_test, '--', label='actual')
plt.plot(best_model(x_test).detach().numpy(), label='FCNN')
plt.plot(hwes_predictor(x_test).detach().numpy(), label='HWES')
plt.legend()
plt.savefig('test_comparison.png')
plt.close()

test_n = len(y_test)
net_abs_dev = (best_model(x_test) - y_test).abs_()
hwes_abs_dev = (hwes_predictor(x_test) - y_test).abs_()
diff_pos = F.relu(hwes_abs_dev - net_abs_dev).reshape(test_n).detach().numpy()
diff_min = (-F.relu(net_abs_dev - hwes_abs_dev)).reshape(test_n).detach().numpy()

plt.title('HWES Predictor VS FCNN Predictor')
plt.hlines(0, xmin=0, xmax=test_n, linestyles='dashed')
plt.bar(list(range(test_n)), diff_pos, color='g', label='FCNN Wins')
plt.bar(list(range(test_n)), diff_min, color='r', label='HWES Wins')
plt.legend()
plt.savefig('predictor_comparison.png')
plt.close()

print("\nGraphiques sauvegardés!")