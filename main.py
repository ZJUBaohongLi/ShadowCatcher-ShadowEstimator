import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
from utils.wasserstein_distance import SinkhornDistance
from models.nn_generator import NNGenerator
from models.z_estimator import ZEstimator
from models.regression_bnn import RegressionBNN
from models.q_func import QFunc
from models.selection_net import SelectionNet
from models.z_net import ZNet
from models.or_net import ORNet
import pandas as pd
from config import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GeneratorDataset(Dataset):
    def __init__(self, t, x, y, s):
        self.t = t
        self.x = x
        self.y = y
        self.s = s

    def __getitem__(self, item):
        return self.t[item], self.x[item], self.y[item], self.s[item]

    def __len__(self):
        return len(self.t)


class EstimatorDataset(Dataset):
    def __init__(self, t, x, y):
        self.t = t
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.t[item], self.x[item], self.y[item]

    def __len__(self):
        return len(self.t)


class ZDataset(Dataset):
    def __init__(self, t, x, z, s):
        self.t = t
        self.x = x
        self.z = z
        self.s = s

    def __getitem__(self, item):
        return self.t[item], self.x[item], self.z[item], self.s[item]

    def __len__(self):
        return len(self.t)


class ORDataset(Dataset):
    def __init__(self, t, x, y, or_tilde):
        self.t = t
        self.x = x
        self.y = y
        self.or_tilde = or_tilde

    def __getitem__(self, item):
        return self.t[item], self.x[item], self.y[item], self.or_tilde[item]

    def __len__(self):
        return len(self.t)


class SelectionDataset(Dataset):
    def __init__(self, t, x, z, s):
        self.t = t
        self.x = x
        self.z = z
        self.s = s

    def __getitem__(self, item):
        return self.t[item], self.x[item], self.z[item], self.s[item]

    def __len__(self):
        return len(self.t)


def load_data(file_path):
    data = pd.read_csv(file_path)
    t = np.array(data['X1']).reshape((-1, 1))
    s = np.array(data['S1']).reshape((-1, 1))
    y = np.array(data['Y']).reshape((-1, 1))
    x = np.array(data[['X2_' + str(i) for i in range(1, config.get('confounds_num') + 1)]])
    gt = np.array(data['GT']).reshape((-1, 1))
    t_train, t_test, x_train, x_test, y_train, y_test, s_train, s_test, gt_train, gt_test = train_test_split(
        t, x, y, s, gt, test_size=0.40)
    t_test, t_val, x_test, x_val, y_test, y_val, s_test, s_val, gt_test, gt_val = train_test_split(
        t_test, x_test, y_test, s_test, gt_test, test_size=0.50)
    return t_train, t_test, t_val, x_train, x_test, x_val, y_train, y_test, y_val, s_train, s_test, s_val, gt_train, gt_test, gt_val


def build_generator_dataset(t_train, t_test, t_val, x_train, x_test, x_val, y_train, y_test, y_val,
                            t_train_us, t_test_us, t_val_us, x_train_us, x_test_us, x_val_us):
    generator_train_dataset = GeneratorDataset(np.concatenate((t_train, t_train_us), 0),
                                               np.concatenate((x_train, x_train_us), 0),
                                               np.concatenate((y_train, np.zeros(t_train_us.shape)), 0),
                                               np.concatenate((np.ones(t_train.shape), np.zeros(t_train_us.shape)), 0))
    generator_test_dataset = GeneratorDataset(np.concatenate((t_test, t_test_us), 0),
                                              np.concatenate((x_test, x_test_us), 0),
                                              np.concatenate((y_test, np.zeros(t_test_us.shape)), 0),
                                              np.concatenate((np.ones(t_test.shape), np.zeros(t_test_us.shape)), 0))
    generator_val_dataset = GeneratorDataset(np.concatenate((t_val, t_val_us), 0),
                                             np.concatenate((x_val, x_val_us), 0),
                                             np.concatenate((y_val, np.zeros(t_val_us.shape)), 0),
                                             np.concatenate((np.ones(t_val.shape), np.zeros(t_val_us.shape)), 0))
    return generator_train_dataset, generator_test_dataset, generator_val_dataset


def build_z_dataset(t_train, t_test, t_val, x_train, x_test, x_val, z_train, z_test, z_val,
                    t_train_us, t_test_us, t_val_us, x_train_us, x_test_us, x_val_us, z_train_us, z_test_us, z_val_us):
    train_dataset = ZDataset(np.concatenate((t_train, t_train_us), 0),
                             np.concatenate((x_train, x_train_us), 0),
                             np.concatenate((z_train, z_train_us), 0),
                             np.concatenate((np.ones(t_train.shape), np.zeros(t_train_us.shape)), 0))
    test_dataset = ZDataset(np.concatenate((t_test, t_test_us), 0),
                            np.concatenate((x_test, x_test_us), 0),
                            np.concatenate((z_test, z_test_us), 0),
                            np.concatenate((np.ones(t_test.shape), np.zeros(t_test_us.shape)), 0))
    val_dataset = ZDataset(np.concatenate((t_val, t_val_us), 0),
                           np.concatenate((x_val, x_val_us), 0),
                           np.concatenate((z_val, z_val_us), 0),
                           np.concatenate((np.ones(t_val.shape), np.zeros(t_val_us.shape)), 0))
    return train_dataset, test_dataset, val_dataset


def build_or_dataset(t_train, t_test, t_val, x_train, x_test, x_val, y_train, y_test, y_val, or_tilde_train,
                     or_tilde_test, or_tilde_val):
    train_dataset = ORDataset(t_train, x_train, y_train, or_tilde_train)
    test_dataset = ORDataset(t_test, x_test, y_test, or_tilde_test)
    val_dataset = ORDataset(t_val, x_val, y_val, or_tilde_val)
    return train_dataset, test_dataset, val_dataset


def build_selection_dataset(t_train, t_test, t_val, x_train, x_test, x_val, z_train, z_test, z_val, s_train,
                            s_test, s_val):
    train_dataset = SelectionDataset(t_train, x_train, z_train, s_train)
    test_dataset = SelectionDataset(t_test, x_test, z_test, s_test)
    val_dataset = SelectionDataset(t_val, x_val, z_val, s_val)
    return train_dataset, test_dataset, val_dataset


def build_estimator_dataset(t_train, t_test, t_val, x_train, x_test, x_val, y_train, y_test, y_val):
    train_dataset = EstimatorDataset(t_train, x_train, y_train)
    test_dataset = EstimatorDataset(t_test, x_test, y_test)
    val_dataset = EstimatorDataset(t_val, x_val, y_val)
    return train_dataset, test_dataset, val_dataset


def train_nn_generator(model, estimator, dataloader, val_dataloader):
    max_iter = config.get('generator_max_iter')
    threshold = config.get('generator_threshold')
    epochs = config.get('generator_epochs')
    q_epochs = config.get('q_epochs')
    lr_e = config.get('estimator_lr')
    wd_e = config.get('estimator_wd')
    lr_g = config.get('generator_lr')
    wd_g = config.get('generator_wd')
    lr_s = config.get('selection_lr')
    wd_s = config.get('selection_wd')
    lr_q = config.get('q_lr')
    wd_q = config.get('q_wd')
    selection_model = ZEstimator(config).to(device)
    q_model = QFunc(config).to(device)
    opt_e = torch.optim.Adam(estimator.parameters(), lr=lr_e, weight_decay=wd_e)
    opt_g = torch.optim.Adam(model.parameters(), lr=lr_g, weight_decay=wd_g)
    opt_s = torch.optim.Adam(selection_model.parameters(), lr=lr_s, weight_decay=wd_s)
    opt_q = torch.optim.Adam(q_model.parameters(), lr=lr_q, weight_decay=wd_q)
    mse_func = nn.MSELoss(reduction='sum')
    wasserstein_func = SinkhornDistance(0.1, 100, reduction='mean', device=device)
    writer = SummaryWriter()
    for iter in range(max_iter):
        rej_flag = True
        for epoch in range(epochs):
            estimator.train()
            model.train()
            selection_model.train()
            loss_e_sum = 0
            loss_s_sum = 0
            loss_g_sum = 0
            for batch in dataloader:
                tall, xall, yall, sall = batch
                tall = tall.to(device)
                xall = xall.to(device)
                yall = yall.to(device)
                sall = sall.to(device)
                sall = torch.squeeze(sall)
                t = tall[sall == 1]
                x = xall[sall == 1]
                y = yall[sall == 1]
                tus = tall[sall == 0]
                xus = xall[sall == 0]
                z = model(x)
                y_hat, t_rep, c_rep = estimator(t, torch.cat((x, z), 1))
                y_minus, _, _ = estimator(t, torch.cat((torch.rand(x.size()).to(device), z), 1))
                loss_y = torch.mean(mse_func(y.float(), y_hat))
                loss_ipm, _, _ = wasserstein_func(t_rep, c_rep)
                loss_e = loss_y + config.get('ipm_weight') * loss_ipm + torch.mean(
                    mse_func(y.float(), y_minus))
                opt_e.zero_grad()
                loss_e.backward()
                opt_e.step()
                loss_e_sum += loss_e
                z = model(x)
                z_hat = selection_model(torch.cat((x, t, y), 1))
                loss_s = torch.mean(mse_func(z.float(), z_hat))
                opt_s.zero_grad()
                loss_s.backward()
                opt_s.step()
                loss_s_sum += loss_s
                z = model(x)
                zus = model(xus)
                y_hat, _, _ = estimator(t, torch.cat((x, z), 1))
                y_minus, _, _ = estimator(t, torch.cat((x, torch.rand(z.size()).to(device)), 1))
                loss_y = torch.mean(mse_func(y.float(), y_hat)) + torch.mean(
                    -mse_func(y.float(), y_minus))
                yus_hat, _, _ = estimator(tus, torch.cat((xus, zus), 1))
                zus_hat = selection_model(torch.cat((xus, tus, yus_hat), 1))
                loss_z = torch.mean(mse_func(zus, zus_hat))
                loss_g = loss_y + loss_z
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                loss_g_sum += loss_g
            writer.add_scalar('Estimator Train loss', loss_e_sum, epoch)
            writer.add_scalar('Generator Train loss', loss_g_sum, epoch)
            writer.add_scalar('Selection Train loss', loss_s_sum, epoch)
            estimator.eval()
            model.eval()
            selection_model.eval()
            loss_e_sum = 0
            loss_s_sum = 0
            loss_g_sum = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    tall, xall, yall, sall = batch
                    tall = tall.to(device)
                    xall = xall.to(device)
                    yall = yall.to(device)
                    sall = sall.to(device)
                    sall = torch.squeeze(sall)
                    t = tall[sall == 1]
                    x = xall[sall == 1]
                    y = yall[sall == 1]
                    tus = tall[sall == 0]
                    xus = xall[sall == 0]
                    z = model(x)
                    y_hat, t_rep, c_rep = estimator(t, torch.cat((x, z), 1))
                    y_minus, _, _ = estimator(t, torch.cat((torch.rand(x.size()).to(device), z), 1))
                    loss_y = torch.mean(mse_func(y.float(), y_hat))
                    loss_ipm, _, _ = wasserstein_func(t_rep, c_rep)
                    loss_e = loss_y + config.get('ipm_weight') * loss_ipm + torch.mean(
                        mse_func(y.float(), y_minus))
                    z_hat = selection_model(torch.cat((x, t, y), 1))
                    loss_e_sum += loss_e
                    z = model(x)
                    loss_s = torch.mean(mse_func(z.float(), z_hat))
                    loss_s_sum += loss_s
                    z = model(x)
                    zus = model(xus)
                    y_hat, _, _ = estimator(t, torch.cat((x, z), 1))
                    y_minus, _, _ = estimator(t, torch.cat((x, torch.rand(z.size()).to(device)), 1))
                    loss_y = torch.mean(mse_func(y.float(), y_hat)) + torch.mean(
                        -mse_func(y.float(), y_minus))
                    yus_hat, _, _ = estimator(tus, torch.cat((xus, zus), 1))
                    zus_hat = selection_model(torch.cat((xus, tus, yus_hat), 1))
                    loss_z = torch.mean(mse_func(zus, zus_hat))
                    loss_g = loss_y + loss_z
                    loss_g_sum += loss_g
            writer.add_scalar('Estimator Val loss', loss_e_sum, epoch)
            writer.add_scalar('Generator Val loss', loss_g_sum, epoch)
            writer.add_scalar('Selection Val loss', loss_s_sum, epoch)
        for epoch in range(q_epochs):
            q_model.train()
            loss_q_sum = 0
            for batch in dataloader:
                tall, xall, yall, sall = batch
                tall = tall.to(device)
                xall = xall.to(device)
                yall = yall.to(device)
                sall = sall.to(device)
                sall = torch.squeeze(sall)
                t = tall[sall == 1]
                x = xall[sall == 1]
                y = yall[sall == 1]
                tus = tall[sall == 0]
                xus = xall[sall == 0]
                z = model(x)
                zus = model(xus)
                q_hat = q_model(torch.cat((x, t, y), 1))
                loss_q = torch.mean((1 / (q_hat + 1e-4) - 1) * torch.cat((x, t, z), 1)) - torch.mean(
                    torch.cat((xus, tus, zus), 1))
                loss_q = torch.sqrt(loss_q * loss_q)
                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()
                loss_q_sum += loss_q
            writer.add_scalar('Q Train loss', loss_q_sum, epoch)
            q_model.eval()
            loss_q_sum = 0
            with torch.no_grad():
                for index, batch in enumerate(val_dataloader):
                    tall, xall, yall, sall = batch
                    tall = tall.to(device)
                    xall = xall.to(device)
                    yall = yall.to(device)
                    sall = sall.to(device)
                    sall = torch.squeeze(sall)
                    t = tall[sall == 1]
                    x = xall[sall == 1]
                    y = yall[sall == 1]
                    tus = tall[sall == 0]
                    xus = xall[sall == 0]
                    z = model(x)
                    zus = model(xus)
                    q_hat = q_model(torch.cat((x, t, y), 1))
                    loss_q = torch.mean((1 / (q_hat + 1e-4) - 1) * torch.cat((x, t, z), 1)) - torch.mean(
                        torch.cat((xus, tus, zus), 1))
                    loss_q = torch.sqrt(loss_q * loss_q)
                    loss_q_sum += loss_q
            writer.add_scalar('Q Val loss', loss_q_sum, epoch)
            if loss_q_sum / (index + 1) <= threshold:
                rej_flag = False
                break
        if rej_flag is False:
            break
        threshold = threshold / (iter + 2)
    writer.close()
    return model, estimator


def generate_zs(generator, data):
    generator.eval()
    with torch.no_grad():
        data = data.to(device)
        gen_zs = generator(data).detach().cpu().numpy()
    return gen_zs.reshape(-1, config.get('generator_dim_latent'))


def train_z_net(z_net, z_net_us, train_dataloader, val_dataloader):
    epochs = config.get('z_epochs')
    lr = config.get('z_lr')
    wd = config.get('z_wd')
    opt = torch.optim.Adam(z_net.parameters(), lr=lr, weight_decay=wd)
    opt_us = torch.optim.Adam(z_net_us.parameters(), lr=lr, weight_decay=wd)
    mse_func = nn.MSELoss(reduction='sum')
    writer = SummaryWriter()
    for epoch in range(epochs):
        z_net.train()
        z_net_us.train()
        loss_sum = 0
        loss_sum_us = 0
        for batch in train_dataloader:
            tall, xall, zall, sall = batch
            tall = tall.to(device)
            xall = xall.to(device)
            zall = zall.to(device)
            sall = sall.to(device)
            sall = torch.squeeze(sall)
            t = tall[sall == 1]
            x = xall[sall == 1]
            z = zall[sall == 1]
            tus = tall[sall == 0]
            xus = xall[sall == 0]
            zus = zall[sall == 0]
            z_hat = z_net(torch.cat((t, x), 1))
            loss = torch.mean(mse_func(z.float(), z_hat))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss
            zus_hat = z_net_us(torch.cat((tus, xus), 1))
            loss_us = torch.mean(mse_func(zus.float(), zus_hat))
            opt_us.zero_grad()
            loss_us.backward()
            opt_us.step()
            loss_sum_us += loss_us
        writer.add_scalar('ZNet Train loss', loss_sum, epoch)
        writer.add_scalar('ZNetUS Train loss', loss_sum_us, epoch)
        z_net.eval()
        z_net_us.eval()
        loss_sum = 0
        loss_sum_us = 0
        with torch.no_grad():
            for batch in val_dataloader:
                tall, xall, zall, sall = batch
                tall = tall.to(device)
                xall = xall.to(device)
                zall = zall.to(device)
                sall = sall.to(device)
                sall = torch.squeeze(sall)
                t = tall[sall == 1]
                x = xall[sall == 1]
                z = zall[sall == 1]
                tus = tall[sall == 0]
                xus = xall[sall == 0]
                zus = zall[sall == 0]
                z_hat = z_net(torch.cat((t, x), 1))
                loss = torch.mean(mse_func(z.float(), z_hat))
                loss_sum += loss
                zus_hat = z_net_us(torch.cat((tus, xus), 1))
                loss_us = torch.mean(mse_func(zus.float(), zus_hat))
                loss_sum_us += loss_us
        writer.add_scalar('ZNet Val loss', loss_sum, epoch)
        writer.add_scalar('ZNetUS Val loss', loss_sum_us, epoch)
    writer.close()
    return z_net, z_net_us


def predict_or_tilde(z_net, z_net_us, data):
    z_net.eval()
    z_net_us.eval()
    with torch.no_grad():
        data = data.to(device)
        gen_zs = z_net(data)
        gen_zus = z_net_us(data)
        or_tilde = gen_zs / (gen_zus + 1e-4)
        or_tilde = torch.mean(or_tilde, dim=1).detach().cpu().numpy()
    return or_tilde.reshape(-1, 1)


def train_or_net(or_net, train_dataloader, val_dataloader):
    epochs = config.get('or_epochs')
    lr = config.get('or_lr')
    wd = config.get('or_wd')
    opt = torch.optim.Adam(or_net.parameters(), lr=lr, weight_decay=wd)
    mse_func = nn.MSELoss(reduction='sum')
    writer = SummaryWriter()
    for epoch in range(epochs):
        or_net.train()
        loss_sum = 0
        for batch in train_dataloader:
            t, x, y, or_tilde = batch
            t = t.to(device)
            x = x.to(device)
            y = y.to(device)
            or_tilde = or_tilde.to(device)
            or_tilde_hat = or_net(torch.cat((t, x, y), 1))
            loss = torch.mean(mse_func(or_tilde.float(), or_tilde_hat))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss
        writer.add_scalar('ORNet Train loss', loss_sum, epoch)
        or_net.eval()
        loss_sum = 0
        with torch.no_grad():
            for batch in val_dataloader:
                t, x, y, or_tilde = batch
                t = t.to(device)
                x = x.to(device)
                y = y.to(device)
                or_tilde = or_tilde.to(device)
                or_tilde_hat = or_net(torch.cat((t, x, y), 1))
                loss = torch.mean(mse_func(or_tilde.float(), or_tilde_hat))
                loss_sum += loss
        writer.add_scalar('ORNet Val loss', loss_sum, epoch)
    writer.close()
    return or_net


def predict_or_hat(or_net, t, x, y):
    or_net.eval()
    with torch.no_grad():
        t = t.to(device)
        x = x.to(device)
        y = y.to(device)
        gen_or_tilde = or_net(torch.cat((t, x, y), 1))
        gen_or_tilde0 = or_net(torch.cat((t, x, torch.zeros(y.shape, dtype=torch.float32).to(device)), 1))
        or_hat = gen_or_tilde / (gen_or_tilde0 + 1e-4)
        or_hat = or_hat.detach().cpu().numpy()
    return or_hat.reshape(-1, 1)


def train_selection_net(selection_net, train_dataloader, val_dataloader):
    epochs = config.get('selection_epochs')
    lr = config.get('selectionnet_lr')
    wd = config.get('selectionnet_wd')
    opt = torch.optim.Adam(selection_net.parameters(), lr=lr, weight_decay=wd)
    bce_func = nn.BCELoss(reduction='sum')
    writer = SummaryWriter()
    for epoch in range(epochs):
        selection_net.train()
        loss_sum = 0
        for batch in train_dataloader:
            t, x, z, s = batch
            t = t.to(device)
            x = x.to(device)
            z = z.to(device)
            s = s.to(device)
            s_hat = selection_net(torch.cat((t, x, z), 1))
            loss = torch.mean(bce_func(s_hat, s.float()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss
        writer.add_scalar('SelectionNet Train loss', loss_sum, epoch)
        selection_net.eval()
        loss_sum = 0
        with torch.no_grad():
            for batch in val_dataloader:
                t, x, z, s = batch
                t = t.to(device)
                x = x.to(device)
                z = z.to(device)
                s = s.to(device)
                s_hat = selection_net(torch.cat((t, x, z), 1))
                loss = torch.mean(bce_func(s_hat, s.float()))
                loss_sum += loss
        writer.add_scalar('SelectionNet Val loss', loss_sum, epoch)
    writer.close()
    return selection_net


def get_weight(selection_net, t, x, z):
    selection_net.eval()
    with torch.no_grad():
        t = t.to(device)
        x = x.to(device)
        z = z.to(device)
        p_hat = selection_net(torch.cat((t, x, z), 1))
        p_hat = p_hat.detach().cpu().numpy()
    return p_hat.reshape(-1, 1)


def train_estimator_bnn(estimator, train_dataloader, val_dataloader):
    epochs = config.get('estimator_epochs')
    lr = config.get('estimator_lr')
    weight_decay = config.get('estimator_wd')
    ipm_weight = config.get('ipm_weight')
    regression_loss_func = torch.nn.MSELoss(reduction='mean')
    wasserstein_func = SinkhornDistance(0.1, 100, reduction='mean', device=device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr, weight_decay=weight_decay)
    writer = SummaryWriter()
    for epoch in range(epochs):
        loss_sum = 0
        estimator.train()
        for index, batch in enumerate(train_dataloader):
            t, x, ground_truth = batch
            t = t.to(device)
            x = x.to(device)
            ground_truth = ground_truth.to(device)
            y_pre, t_rep, c_rep = estimator(t, x)
            loss1 = regression_loss_func(y_pre.to(torch.float32), ground_truth.to(torch.float32))
            loss2, _, _ = wasserstein_func(t_rep, c_rep)
            loss = loss1 + ipm_weight * loss2
            loss_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar('Unselected Estimator Train loss', loss_sum, epoch)
        loss_sum = 0
        estimator.eval()
        with torch.no_grad():
            for index, batch in enumerate(val_dataloader):
                t, x, ground_truth = batch
                t = t.to(device)
                x = x.to(device)
                ground_truth = ground_truth.to(device)
                y_pre, t_rep, c_rep = estimator(t, x)
                loss1 = regression_loss_func(y_pre.to(torch.float32), ground_truth.to(torch.float32))
                loss = loss1
                loss_sum += loss
        writer.add_scalar('Unselected Estimator Val loss', loss_sum, epoch)
    writer.close()
    return estimator


def test_estimator_bnn(estimator, t, x):
    estimator.eval()
    with torch.no_grad():
        t1 = torch.tensor(t, dtype=torch.float32).to(device)
        x1 = torch.tensor(x, dtype=torch.float32).to(device)
        y_pre, _, _ = estimator(t1, x1)
        y_pre = y_pre.detach().cpu().numpy()
    return y_pre.reshape((-1, 1))


res_list = []
for i in range(config.get('experiment_num')):
    print("Data Preparation")
    obs_file_path = 'data/ihdp_selection_data.csv'
    unselected_file_path = 'data/ihdp_unselection_data.csv'
    print("ShadowCatcher")
    obs_t_train, obs_t_test, obs_t_val, obs_x_train, obs_x_test, obs_x_val, obs_y_train, obs_y_test, obs_y_val, obs_s_train, obs_s_test, obs_s_val, obs_gt_train, obs_gt_test, obs_gt_val = load_data(
        obs_file_path)
    unselected_t_train, unselected_t_test, unselected_t_val, unselected_x_train, unselected_x_test, unselected_x_val, unselected_y_train, unselected_y_test, unselected_y_val, unselected_s_train, unselected_s_test, unselected_s_val, unselected_gt_train, unselected_gt_test, unselected_gt_val = load_data(
        unselected_file_path)
    generator_train_dataset, generator_test_dataset, generator_val_dataset = build_generator_dataset(
        obs_t_train, obs_t_test, obs_t_val,
        obs_x_train, obs_x_test, obs_x_val,
        obs_y_train, obs_y_test, obs_y_val,
        unselected_t_train, unselected_t_test, unselected_t_val,
        unselected_x_train, unselected_x_test, unselected_x_val
    )
    gen_train_dataloader = DataLoader(generator_train_dataset, batch_size=config.get('gen_batch_num'), shuffle=True,
                                      drop_last=False)
    gen_val_dataloader = DataLoader(generator_val_dataset, batch_size=config.get('gen_batch_num'), shuffle=True,
                                    drop_last=False)
    generator = NNGenerator(config).to(device)
    estimator = RegressionBNN(config).to(device)
    generator, estimator = train_nn_generator(generator, estimator, gen_train_dataloader, gen_val_dataloader)
    obs_z_train = generate_zs(generator, torch.tensor(obs_x_train, dtype=torch.float32))
    obs_z_val = generate_zs(generator, torch.tensor(obs_x_val, dtype=torch.float32))
    obs_z_test = generate_zs(generator, torch.tensor(obs_x_test, dtype=torch.float32))
    unselected_z_train = generate_zs(generator, torch.tensor(unselected_x_train, dtype=torch.float32))
    unselected_z_val = generate_zs(generator, torch.tensor(unselected_x_val, dtype=torch.float32))
    unselected_z_test = generate_zs(generator, torch.tensor(unselected_x_test, dtype=torch.float32))
    print('ShadowEstimator')
    z_train_dataset, z_test_dataset, z_val_dataset = build_z_dataset(
        obs_t_train, obs_t_test, obs_t_val,
        obs_x_train, obs_x_test, obs_x_val,
        obs_z_train, obs_z_test, obs_z_val,
        unselected_t_train, unselected_t_test, unselected_t_val,
        unselected_x_train, unselected_x_test, unselected_x_val,
        unselected_z_train, unselected_z_test, unselected_z_val
    )
    z_train_dataloader = DataLoader(z_train_dataset, batch_size=config.get('z_batch_num'), shuffle=True,
                                    drop_last=False)
    z_val_dataloader = DataLoader(z_val_dataset, batch_size=config.get('z_batch_num'), shuffle=True,
                                  drop_last=False)
    z_net = ZNet(config).to(device)
    z_net_us = ZNet(config).to(device)
    z_net, z_net_us = train_z_net(z_net, z_net_us, z_train_dataloader, z_val_dataloader)
    obs_or_tilde_train = predict_or_tilde(z_net, z_net_us,
                                          torch.tensor(np.concatenate((obs_t_train, obs_x_train), axis=1),
                                                       dtype=torch.float32))
    obs_or_tilde_test = predict_or_tilde(z_net, z_net_us,
                                         torch.tensor(np.concatenate((obs_t_test, obs_x_test), axis=1),
                                                      dtype=torch.float32))
    obs_or_tilde_val = predict_or_tilde(z_net, z_net_us, torch.tensor(np.concatenate((obs_t_val, obs_x_val), axis=1),
                                                                      dtype=torch.float32))
    or_train_dataset, or_test_dataset, or_val_dataset = build_or_dataset(
        obs_t_train, obs_t_test, obs_t_val,
        obs_x_train, obs_x_test, obs_x_val,
        obs_y_train, obs_y_test, obs_y_val,
        obs_or_tilde_train, obs_or_tilde_test, obs_or_tilde_val
    )
    or_train_dataloader = DataLoader(or_train_dataset, batch_size=config.get('or_batch_num'), shuffle=True,
                                     drop_last=False)
    or_val_dataloader = DataLoader(or_val_dataset, batch_size=config.get('or_batch_num'), shuffle=True,
                                   drop_last=False)
    or_net = ORNet(config).to(device)
    or_net = train_or_net(or_net, or_train_dataloader, or_val_dataloader)
    obs_or_hat_train = predict_or_hat(or_net, torch.tensor(obs_t_train, dtype=torch.float32),
                                      torch.tensor(obs_x_train, dtype=torch.float32),
                                      torch.tensor(obs_y_train, dtype=torch.float32))
    obs_or_hat_test = predict_or_hat(or_net, torch.tensor(obs_t_test, dtype=torch.float32),
                                     torch.tensor(obs_x_test, dtype=torch.float32),
                                     torch.tensor(obs_y_test, dtype=torch.float32))
    obs_or_hat_val = predict_or_hat(or_net, torch.tensor(obs_t_val, dtype=torch.float32),
                                    torch.tensor(obs_x_val, dtype=torch.float32),
                                    torch.tensor(obs_y_val, dtype=torch.float32))
    obs_y_hat_train = test_estimator_bnn(estimator, obs_t_train, np.concatenate((obs_x_train, obs_z_train), axis=1))
    obs_y_hat_test = test_estimator_bnn(estimator, obs_t_test, np.concatenate((obs_x_test, obs_z_test), axis=1))
    obs_y_hat_val = test_estimator_bnn(estimator, obs_t_val, np.concatenate((obs_x_val, obs_z_val), axis=1))
    obs_y_hat_test_cf = test_estimator_bnn(estimator, 1 - obs_t_test,
                                           np.concatenate((obs_x_test, obs_z_test), axis=1))
    unselected_y_hat_test = test_estimator_bnn(estimator, unselected_t_test,
                                               np.concatenate((unselected_x_test, unselected_z_test), axis=1))
    unselected_y_hat_test_cf = test_estimator_bnn(estimator, 1 - unselected_t_test,
                                                  np.concatenate((unselected_x_test, unselected_z_test), axis=1))
    obs_mean_or_train = predict_or_hat(or_net, torch.tensor(obs_t_train, dtype=torch.float32),
                                       torch.tensor(obs_x_train, dtype=torch.float32),
                                       torch.tensor(obs_y_hat_train, dtype=torch.float32))
    obs_mean_or_test = predict_or_hat(or_net, torch.tensor(obs_t_test, dtype=torch.float32),
                                      torch.tensor(obs_x_test, dtype=torch.float32),
                                      torch.tensor(obs_y_hat_test, dtype=torch.float32))
    obs_mean_or_val = predict_or_hat(or_net, torch.tensor(obs_t_val, dtype=torch.float32),
                                     torch.tensor(obs_x_val, dtype=torch.float32),
                                     torch.tensor(obs_y_hat_val, dtype=torch.float32))
    obs_unselected_y_hat_train = obs_or_hat_train * obs_y_hat_train / (obs_mean_or_train + 1e-4)
    obs_unselected_y_hat_test = obs_or_hat_test * obs_y_hat_test / (obs_mean_or_test + 1e-4)
    obs_unselected_y_hat_val = obs_or_hat_val * obs_y_hat_val / (obs_mean_or_val + 1e-4)
    unselected_estimator = RegressionBNN(config).to(device)
    obs_unselected_estimator_train_dataset, obs_unselected_estimator_test_dataset, obs_unselected_estimator_val_dataset \
        = build_estimator_dataset(obs_t_train, obs_t_test, obs_t_val,
                                  np.concatenate((obs_x_train, obs_z_train), axis=1),
                                  np.concatenate((obs_x_test, obs_z_test), axis=1),
                                  np.concatenate((obs_x_val, obs_z_val), axis=1),
                                  obs_unselected_y_hat_train, obs_unselected_y_hat_test, obs_unselected_y_hat_val)
    obs_unselected_estimator_train_dataloader = DataLoader(obs_unselected_estimator_train_dataset,
                                                           batch_size=config.get('estimator_batch_num'), shuffle=True,
                                                           drop_last=False)
    obs_unselected_estimator_val_dataloader = DataLoader(obs_unselected_estimator_val_dataset,
                                                         batch_size=config.get('estimator_batch_num'), shuffle=True,
                                                         drop_last=False)
    obs_unselected_estimator = RegressionBNN(config).to(device)
    obs_unselected_estimator = train_estimator_bnn(
        obs_unselected_estimator, obs_unselected_estimator_train_dataloader, obs_unselected_estimator_val_dataloader
    )
    obs_unselected_y_test = test_estimator_bnn(obs_unselected_estimator, obs_t_test,
                                               np.concatenate((obs_x_test, obs_z_test), axis=1))
    obs_unselected_y_test_cf = test_estimator_bnn(obs_unselected_estimator, 1 - obs_t_test,
                                                  np.concatenate((obs_x_test, obs_z_test), axis=1))
    unselected_unselected_y_test = test_estimator_bnn(
        obs_unselected_estimator, unselected_t_test, np.concatenate((unselected_x_test, unselected_z_test), axis=1)
    )
    unselected_unselected_y_test_cf = test_estimator_bnn(
        obs_unselected_estimator, 1 - unselected_t_test, np.concatenate((unselected_x_test, unselected_z_test), axis=1)
    )
    selection_train_dataset, selection_test_dataset, selection_val_dataset = build_selection_dataset(
        np.concatenate((obs_t_train, unselected_t_train), axis=0),
        np.concatenate((obs_t_test, unselected_t_test), axis=0),
        np.concatenate((obs_t_val, unselected_t_val), axis=0),
        np.concatenate((obs_x_train, unselected_x_train), axis=0),
        np.concatenate((obs_x_test, unselected_x_test), axis=0),
        np.concatenate((obs_x_val, unselected_x_val), axis=0),
        np.concatenate((obs_z_train, unselected_z_train), axis=0),
        np.concatenate((obs_z_test, unselected_z_test), axis=0),
        np.concatenate((obs_z_val, unselected_z_val), axis=0),
        np.concatenate(
            (np.ones(obs_t_train.shape, dtype=np.float64), np.zeros(unselected_t_train.shape, dtype=np.float64)),
            axis=0),
        np.concatenate(
            (np.ones(obs_t_test.shape, dtype=np.float64), np.zeros(unselected_t_test.shape, dtype=np.float64)),
            axis=0),
        np.concatenate(
            (np.ones(obs_t_val.shape, dtype=np.float64), np.zeros(unselected_t_val.shape, dtype=np.float64)),
            axis=0)
    )
    selection_train_dataloader = DataLoader(selection_train_dataset, batch_size=config.get('selection_batch_num'),
                                            shuffle=True,
                                            drop_last=False)
    selection_val_dataloader = DataLoader(selection_val_dataset, batch_size=config.get('selection_batch_num'),
                                          shuffle=True,
                                          drop_last=False)
    selection_net = SelectionNet(config).to(device)
    selection_net = train_selection_net(selection_net, selection_train_dataloader, selection_val_dataloader)
    obs_weight_test = get_weight(selection_net, torch.tensor(obs_t_test, dtype=torch.float32),
                                 torch.tensor(obs_x_test, dtype=torch.float32),
                                 torch.tensor(obs_z_test, dtype=torch.float32))
    obs_weight_test_cf = get_weight(selection_net, torch.tensor(1 - obs_t_test, dtype=torch.float32),
                                    torch.tensor(obs_x_test, dtype=torch.float32),
                                    torch.tensor(obs_z_test, dtype=torch.float32))
    obs_unselected_weight_test = 1 - obs_weight_test
    obs_unselected_weight_test_cf = 1 - obs_weight_test_cf
    unselected_weight_test = get_weight(selection_net, torch.tensor(unselected_t_test, dtype=torch.float32),
                                        torch.tensor(unselected_x_test, dtype=torch.float32),
                                        torch.tensor(unselected_z_test, dtype=torch.float32))
    unselected_weight_test_cf = get_weight(selection_net, torch.tensor(1 - unselected_t_test, dtype=torch.float32),
                                           torch.tensor(unselected_x_test, dtype=torch.float32),
                                           torch.tensor(unselected_z_test, dtype=torch.float32))
    unselected_unselected_weight_test = 1 - unselected_weight_test
    unselected_unselected_weight_test_cf = 1 - unselected_weight_test_cf
    obs_y_test_f = obs_weight_test * obs_y_hat_test + obs_unselected_weight_test * obs_unselected_y_test
    obs_y_test_cf = obs_weight_test_cf * obs_y_hat_test_cf + obs_unselected_weight_test_cf * obs_unselected_y_test_cf
    unselected_y_test_f = unselected_weight_test * unselected_y_hat_test + unselected_unselected_weight_test * unselected_unselected_y_test
    unselected_y_test_cf = unselected_weight_test_cf * unselected_y_hat_test_cf + unselected_unselected_weight_test_cf * unselected_unselected_y_test_cf
    ite_test = np.where(obs_t_test == 1, obs_y_test_f - obs_y_test_cf, obs_y_test_cf - obs_y_test_f).reshape(-1, 1)
    unselected_ite_test = np.where(unselected_t_test == 1, unselected_y_test_f - unselected_y_test_cf,
                                   unselected_y_test_cf - unselected_y_test_f).reshape(-1, 1)
    res_list.append(np.sqrt(
        np.mean(np.square(ite_test - obs_gt_test))))
    res_list.append(np.sqrt(
        np.mean(np.square(unselected_ite_test - unselected_gt_test))))
    if device == 'cuda':
        torch.cuda.empty_cache()
res_list = np.array(res_list).reshape(-1, 2)
bias = np.abs(np.mean(res_list, axis=0)).reshape(res_list.shape[1], 1)
sd = np.std(res_list, axis=0).reshape(res_list.shape[1], 1)
np.savetxt('res/result.txt', np.concatenate((bias, sd), 0))
