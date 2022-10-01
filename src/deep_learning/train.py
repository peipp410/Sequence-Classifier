from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy

import util
from model import SeqClassifier


# params
n_random_state = 88
seed = 88
epochs = 200
batch_size = 32
lr = 0.001
weight_decay = 5e-4
dropout = 0.05
device = 'cuda'

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

same_seed(seed)

# Model and optimizer
model = SeqClassifier(dropout=dropout).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

loss_fct = torch.nn.CrossEntropyLoss()

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}

def train(model, data):
    train_loader = DataLoader(data, **params)

    model.train()

    y_pred = []
    y_label = []
    loss_history = 0
    cnt = 0

    for seqs, labels in tqdm(train_loader):
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs)
        loss = loss_fct(logits, labels)
        loss.backward()
        optimizer.step()

        pred_int = torch.argmax(logits, dim=1)
        y_pred = y_pred + pred_int.flatten().tolist()
        y_label = y_label + labels.flatten().tolist()
        loss_history = loss_history + loss.item()

        cnt = cnt + 1

    acc_train = accuracy_score(y_label, y_pred)

    return loss_history / cnt, acc_train


def val_test(model, data):
    val_test_dataloader = DataLoader(data, **params)

    model.eval()

    y_pred = []
    y_label = []
    loss_history = 0
    cnt = 0

    with torch.no_grad():
        for seqs, labels in tqdm(val_test_dataloader):
            seqs, labels = seqs.to(device), labels.to(device)
            logits = model(seqs)
            loss = loss_fct(logits, labels)

            pred_int = torch.argmax(logits, dim=1)
            y_pred = y_pred + pred_int.flatten().tolist()
            y_label = y_label + labels.flatten().tolist()
            loss_history = loss_history + loss.item()

            cnt = cnt + 1

    acc_val_test = accuracy_score(y_label, y_pred)

    return loss_history / cnt, acc_val_test



def main():
    max_acc = 0.3

    # logs
    writer = SummaryWriter('./output/record/logs')

    # 载入数据
    seqs = util.getSeqs('test.fa')
    seqs_array = util.one_hot_encode(seqs)
    y = util.getLabel('seq_id.map')


    x_train, x_test, y_train, y_test = train_test_split(seqs_array, y,
                                                        test_size=0.2, stratify=y, random_state=n_random_state)
    train_data = util.SeqDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_data = util.SeqDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    print('Start Training...')
    for epoch in range(epochs):
        print('-------- Epoch ' + str(epoch + 1) + ' --------')

        loss_train, acc_train = train(model, train_data)
        loss_val, acc_val = val_test(model, val_data)

        print('Training Loss: {:.4f}'.format(loss_train))
        print('Training acc: {:.4f}'.format(acc_train))
        writer.add_scalar('Train Loss', loss_train, global_step=epoch)

        print('Val Loss: {:.4f}'.format(loss_val))
        print('Val acc: {:.4f}'.format(acc_val))
        writer.add_scalar('Val Loss', loss_val, global_step=epoch)

        if acc_val > max_acc:
            model_max = copy.deepcopy(model)
            max_acc = acc_val

            # 保存
            model_max.eval()
            torch.save(model_max, './output/models/model' + str(epoch) + str(max_acc) + '.pth')

    writer.close()



if __name__ == '__main__':
    main()



