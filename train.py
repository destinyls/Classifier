import os
import copy
import time
import json

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from model.model import initialize_model, save_checkpoint
from loss.focal_loss import FocalLoss
from loss.label_smooth_focal_loss import LabelSmoothFocalLoss
from loss.label_smooth import LabelSmoothSoftmaxCEV1

from data.person_dataloader import build_dataloader
from protos import pipeline_pb2
from google.protobuf import text_format
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils.lr_schedule import WarmUpExpLR
from utils.augmentation_utils import mixup_data, mixup_criterion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# criterion = FocalLoss(num_classes = 3, alpha = [0.403, 0.228, 0.368])
criterion = LabelSmoothSoftmaxCEV1(lb_smooth=0.1, ignore_index=-1, reduction='mean')
criterion.cuda()
# criterion = LabelSmoothFocalLoss(num_classes = 4, alpha = [0.119, 0.677, 0.099, 0.104])

def train_fn(model, optimizer, dataloader):
    model.train()
    running_loss, running_corrects = 0.0, 0.0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, 0.1)
            loss_func = mixup_criterion(labels_a, labels_b, lam)     
        # 零参数梯度
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if is_inception and phase == 'train':
                outputs, aux_outputs = model(images)
                if use_mixup:
                    loss1 = loss_func(criterion, outputs)
                    loss2 = loss_func(criterion, aux_outputs)
                    loss = loss1 + 0.4 * loss2
                else:
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
            else:
                outputs = model(images)
                if use_mixup:
                    loss = loss_func(criterion, outputs)
                else:
                    loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            pass
    epoch_loss = running_loss / (len(dataloader) * batch_size)
    epoch_acc = running_corrects.double() / (len(dataloader) * batch_size)
    return epoch_loss, epoch_acc.item()

def val_fn(model, dataloader):
    model.eval()
    running_loss, running_corrects = 0.0, 0.0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            if is_inception:
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            pass
    epoch_loss = running_loss / (len(dataloader) * batch_size)
    epoch_acc = running_corrects.double() / (len(dataloader) * batch_size)

    return epoch_loss, epoch_acc.item()
    

if __name__ == "__main__":
    pipeline_config = "./config/pipeline_config.proto"
    pipeline = pipeline_pb2.PipelineConfig()
    with open(pipeline_config, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline)
    print(pipeline)
    
    data_root = pipeline.data_root
    model_name = pipeline.model.model_name
    num_classes = pipeline.model.num_classes
    num_epochs = pipeline.train_config.num_epochs
    batch_size = pipeline.train_config.batch_size
    checkpoints_path = pipeline.train_config.checkpoints_path
    use_mixup = pipeline.train_config.use_mixup

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
 
    # dataloder
    trainloader, validloader, testloader = build_dataloader(data_root, batch_size=batch_size)
    dataloaders = {"train": trainloader, "val": validloader}
    # model
    model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
    model = model.to(device)
    params_to_update = model.parameters()
    # optimizer
    # optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer = optim.Adam(params_to_update, lr=0.0001, betas=(0.9,0.99))
    # scheduler = ReduceLROnPlateau(optimizer, 'max')
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = WarmUpExpLR(optimizer)

    best_acc = 0.0
    since = time.time()
    is_inception = False
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    valid_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                epoch_loss, epoch_acc = train_fn(model, optimizer, dataloaders["train"])
            else:
                epoch_loss = 0.0
                epoch_acc = 0.0
                val_num = 1
                for i in range(val_num):
                    epo_loss, epo_acc = val_fn(model, dataloaders["val"])
                    epoch_loss += (epo_loss / val_num)
                    epoch_acc += (epo_acc / val_num)
                valid_acc = epoch_acc
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                if epoch_acc > best_acc or True:
                    state = {'epoch': epoch,
                             'optimizer_state_dict': optimizer.state_dict(),
                             'model_state_dict': model.state_dict(),
                             'acc': epoch_acc}
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(checkpoints_path, 'model_{}_{}.pth'.format(state["epoch"], state["acc"])))
                    save_checkpoint(state, filepath=checkpoints_path, is_best=True)

        scheduler.step()
        with open(os.path.join(checkpoints_path, "train_loss.json"),'w') as file_object:
            json.dump(train_loss, file_object)
        with open(os.path.join(checkpoints_path, "val_loss.json"),'w') as file_object:
            json.dump(val_loss, file_object)
        with open(os.path.join(checkpoints_path, "train_acc.json"),'w') as file_object:
            json.dump(train_acc, file_object)
        with open(os.path.join(checkpoints_path, "val_acc.json"),'w') as file_object:
            json.dump(val_acc, file_object)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
