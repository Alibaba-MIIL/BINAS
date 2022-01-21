import copy
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

class Quadratic(torch.nn.Module):
    def __init__(self, in_features):
        super(Quadratic, self).__init__()
        self.linear = nn.Linear(in_features, 1, bias=True)
        self.bilinear = nn.Bilinear(in_features, in_features, 1, bias=False)

    def forward(self, x):
        out = self.linear(x)
        out += self.bilinear(x, x)
        return out


class Bilinear(torch.nn.Module):
    def __init__(self, in1_features, in2_features):
        super(Bilinear, self).__init__()
        self._in1_features = in1_features
        self._in2_features = in2_features

        self.linear = nn.Linear(in1_features + in2_features, 1, bias=True)
        self.bilinear = nn.Bilinear(in1_features, in2_features, 1, bias=False)

    def forward(self, x):
        out = self.linear(x)
        if len(x.shape) > 1:
            a = x[:, :self._in1_features]
            b = x[:, -self._in2_features:]
        else:
            a = x[:self._in1_features]
            b = x[-self._in2_features:]
        out += self.bilinear(a, b)
        return out


class MLP(nn.Module):
    # N-layer MLP
    def __init__(self, n_feature, n_layers=3, n_hidden=500, n_output=1, drop=0.2):
        super(MLP, self).__init__()

        self.stem = nn.Sequential(nn.Linear(n_feature, n_hidden), nn.ReLU())

        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)

        self.regressor = nn.Linear(n_hidden, n_output)  # output layer
        # self.regressor.bias.data.fill_(0.76)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.stem(x)
        x = self.hidden(x)
        x = self.drop(x)
        x = self.regressor(x)  # linear output
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)


class Predictor:
    """ Multi Layer Perceptron """

    def __init__(self, constructor, *args, **kwargs):
        # print(constructor)
        self.model = constructor(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.model = train(self.model, *args, **kwargs)
        return self

    def eval(self):
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        retval = self.model.load_state_dict(state_dict)
        self.model.cuda()
        return retval

    def __call__(self, x):
        x = torch.tensor(x).float().to(device=next(self.model.parameters()).device) \
            if not isinstance(x, torch.Tensor) else x
        retval = self.model(x)
        return retval


def train(net, x, y, trn_split=0.8, pretrained=None, device='cuda',
          lr=1e-6, epochs=30000, weight_decay=1e-5, verbose=True):
    n_samples = x.shape[0]

    target = torch.zeros(n_samples, 1)
    perm = torch.randperm(target.size(0))
    train_size = int(n_samples * trn_split)
    trn_idx = perm[:train_size]
    vld_idx = perm[train_size:]

    inputs = torch.from_numpy(x).float()
    target[:, 0] = torch.from_numpy(y).float()
    m = torch.mean(target, axis=0)
    print("Train set: {}, Test set: {}".format(train_size, n_samples - train_size))
    print("Set Bias to mean acc: {}".format(m[0]))
    if hasattr(net, 'regressor'):
        net.regressor.bias.data.fill_(m[0])

    print("start training...")
    # back-propagation training of a NN
    if pretrained is not None:
        print("Constructing MLP surrogate model with pre-trained weights")
        init = torch.load(pretrained, map_location='cpu')
        net.load_state_dict(init)
        best_net = copy.deepcopy(net)
    # else:
    # print("Constructing MLP surrogate model with "
    #       "sample size = {}, epochs = {}".format(x.shape[0], epochs))

    # initialize the weights
    # net.apply(Net.init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=0)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=1, epochs=int(epochs),
    #                                                 pct_start=0.1)
    best_loss = 1e33
    for epoch in range(epochs):
        trn_inputs = inputs[trn_idx]
        trn_labels = target[trn_idx]
        loss_trn = train_one_epoch(net, trn_inputs, trn_labels, criterion, optimizer, device)
        loss_vld = infer(net, inputs[vld_idx], target[vld_idx], criterion, device)
        scheduler.step()

        if epoch % 500 == 0 and verbose:
            print("Epoch {:4d}: trn loss = {:.4E}, vld loss = {:.4E}".format(epoch, loss_trn, loss_vld))

        # if loss_trn < best_loss:
        #     best_loss = loss_trn
        #     best_net = copy.deepcopy(net)

        if loss_vld < best_loss:
            last_epoch = epoch
            best_loss = loss_vld
            best_net = copy.deepcopy(net)

    # validate(best_net, inputs, target, device=device)
    print(f'Early stopping at epoch {last_epoch}/{epochs}')

    return best_net.cuda()


def train_one_epoch(net, data, target, criterion, optimizer, device):
    net.train()

    data, target = data.to(device), target.to(device)
    pred = net(data)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    return loss.item()


def infer(net, data, target, criterion, device):
    net.eval()

    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        pred = net(data)
        loss = criterion(pred, target)

    return loss.item()


def construct_predictors(predictor_type, alphas, betas):
    type2args = {'mlp': [MLP, alphas + betas],
                 'bilinear': [Bilinear, alphas, betas],
                 'quadratic': [Quadratic, alphas + betas]}
    assert predictor_type in type2args
    return Predictor(*type2args[predictor_type])


def closed_form_solution(predictor, X, Y, svd_truncate=3000):
    is_quad = isinstance(predictor.model, Quadratic)
    assert is_quad or isinstance(predictor.model, Bilinear), \
        'A closed form solution is available only for bilinear and quadratic predictors'
    shape = predictor.model.bilinear.weight.shape
    alphas, betas = shape[-2], shape[-1]
    X2 = torch.cat([x.unsqueeze(0) for x in X], dim=0).float()
    if is_quad:
        X_kron = torch.cat([torch.tensor(np.kron(x, x)).unsqueeze(0) for x in X], dim=0)
        X2 = torch.cat([X_kron, X2], dim=1)
    else:
        X_kron = torch.cat([torch.tensor(np.kron(x[:alphas], x[alphas:])).unsqueeze(0) for x in X], dim=0)
        X2 = torch.cat([X_kron, X2], dim=1)

    Y2 = torch.cat([y.unsqueeze(0) for y in Y], dim=0)
    sx = (torch.sum(X2, dim=0) / float(1.0 * X2.shape[0])).unsqueeze(0)
    sy = (torch.sum(Y2, dim=0) / float(1.0 * Y2.shape[0])).unsqueeze(0)
    X3 = X2 - sx
    Y3 = Y2 - sy
    if is_quad:
        U, s, V = torch.svd_lowrank(X3, q=svd_truncate)
    else:
        U, s, V = torch.svd(X3)

    s2 = 1 / s
    s2[s2 > 10] = 0
    W = (V @ torch.diag(s2) @ U.transpose(0, 1) @ Y3).cpu()
    b = torch.sum(Y2 - X2 @ W) / (1.0 * len(Y2))
    W1 = W[-len(X[0]):]
    W2 = W[:-len(X[0])]
    predictor.model.linear.weight.data = W1.transpose(0, 1)
    predictor.model.linear.bias.data = b * torch.ones_like(predictor.model.linear.bias)
    W2 = W2.reshape(alphas, betas)
    W2 = (W2 + W2.T) / 2 if is_quad else W2
    predictor.model.bilinear.weight.data = W2.unsqueeze(0)
    return predictor


def predict(predictor, X, batch_size=64):
    predictor.eval()
    X = torch.stack(X)
    torch_dataset = Data.TensorDataset(X)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    Y = []
    for step, x in enumerate(loader):
        x = x[0].float().cuda()
        Y.append(predictor(x))

    Y = np.concatenate([y.cpu().detach().numpy() for y in Y]).squeeze().squeeze()
    return Y
