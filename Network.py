import torch.optim as optim
import torch.nn as nn
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Loss
import Scheduler
# from Analysis import show_image_mask, show_image_mask_pred, show_image_tta_pred
from Evaluation import do_kaggle_metric, dice_accuracy, do_mAP, batch_encode, unpad_im

class SegmentationNetwork(nn.Module):
    def __init__(self, lr=0.005, fold=None, val_mode='max', criterion_name='lovasz', comment=''):
        super(SegmentationNetwork, self).__init__()
        self.lr = lr
        self.fold = fold
        self.scheduler = None
        self.best_model_path = None
        self.epoch = 0
        self.val_mode = val_mode
        if criterion_name == 'lovasz':
            self.criterion = Loss.Lovasz_Hinge(per_image=True)
        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf
        self.comment = comment
        self.train_log = dict(loss=[], iou=[], mAP=[])
        self.val_log = dict(loss=[], iou=[], mAP=[])
        self.create_save_folder()
        
    def create_optimizer(self, optimizer='SGD', use_scheduler=None, gamma=0.25, patience=4,
                         milestones=None, T_max=10, T_mul=2, lr_min=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                       lr=self.lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
            
        if use_scheduler == 'ReduceOnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceOnPlateau(self.optimizer, mode='max', factor=gamma,
                                                                patience=patience, verbose=True, threshold=0.01, 
                                                                min_lr=1e-5, eps=1e-8)
        elif use_scheduler == 'Milestones':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma, 
                                                            last_epoch=-1)
        
        elif use_scheduler == 'CosineAnneling':
            self.scheduler = Scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, T_mul=T_mul, lr_min=lr_min, 
                                                         val_mode=self.val_mode,
                                                         last_epoch=-1, save_snapshots=True)
    
    def train_network(self, train_loader, val_loader, n_epoch=10):
        while self.epoch < n_epoch:
            self.epoch += 1
            lr = np.mean([param_group['lr'] for param_group in self.optimizer.param_groups])
            train_loss, train_iou, train_mAP = self.training_step(train_loader)
            val_loss, val_iou, val_mAP = self.perform_validation(val_loader)
            # Learning Rate Scheduler
            if self.scheduler is not None:
                if type(self.scheduler).__name__ == 'ReduceOnPlateau':
                    self.scheduler.step(np.mean(val_mAP))
                elif type(self.scheduler).__name__ == 'CosineAnnealingLR':
                    self.scheduler.step(self.epoch,
                                        save_dict=dict(metric=np.mean(val_mAP),
                                                       save_dir=self.save_dir,
                                                       fold=self.fold,
                                                       state_dict=self.state_dict()))
                else:
                    self.scheduler.step(self.epoch)
            if type(self.scheduler).__name__ != 'CosineAnnealingLR':
                self.save_best_model(np.mean(val_mAP))
            print(('train loss:{:.3f}  val loss:{:.3f} '
                   'train iou: {:.3f}  val iou :{:.3f} '
                   'train mAP: {:.3f}  val mAP :{:.3f} ').format(np.mean(train_loss), np.mean(val_loss),
                                                                 np.mean(train_iou), np.mean(val_iou),
                                                                 np.mean(train_mAP), np.mean(val_mAP)))
            self.save_training_log()
     
    def training_step(self, train_loader):
        self.train()
        train_loss = []
        train_iou = []
        train_mAP = []
        for i, (index, im, mask, z) in enumerate(train_loader):
            self.optimizer.zero_grad()
            im = im.cuda()
            mask = mask.cuda()
            z = z.cuda()
            logit = self.forward(im, z)
            pred = torch.sigmoid(logit)
            loss = self.criterion(logit, mask)
            iou = dice_accuracy(pred, mask, is_average=False)
            mAP = do_mAP(pred.data.cpu().numpy(), mask.cpu().numpy(), is_average=False)
            
            train_loss.append(loss.item())
            train_iou.extend(iou)
            train_mAP.extend(mAP)
            loss.backward()
            self.optimizer.step()
            
        for metric in ['loss', 'iou', 'mAP']:
            self.train_log[metric].append(np.mean(eval('train_{}'.format(metric))))
        return train_loss, train_iou, train_mAP
    
    def perform_validation(self, val_loader):
        self.eval()
        val_loss = []
        val_iou = []
        val_mAP = []
        for i, (index, im, mask, z) in enumerate(val_loader):
            im = im.cuda()
            mask = mask.cuda()
            z = z.cuda()
            
            with torch.no_grad():
                logit = self.forward(im, z)
                pred = torch.sigmoid(logit)
                loss = self.criterion(logit, mask)
                iou = dice_accuracy(pred, mask, is_average=False)
                mAP = do_mAP(pred.cpu().numpy(), mask.cpu().numpy(), is_average=False)
            val_loss.append(loss.item())
            val_iou.extend(iou)
            val_mAP.extend(mAP)
            
        for metric in ['loss', 'iou', 'mAP']:
            self.val_log[metric].append(np.mean(eval('val_{}'.format(metric))))
        return val_loss, val_iou, val_mAP
    
    def predict(self, test_loader, return_rle=False, tta_transform=None, threshold=0.45):
        self.eval()
        self.cuda()
        for i, (index, im, z) in enumerate(test_loader):
            with torch.no_grad():
                im = im.cuda()
                z = z.cuda()
                batch_pred = []
                # TTA
                if tta_transform is not None:
                    tta_list = torch.FloatTensor(tta_transform(im.cpu().numpy(), mode='in'))
                    tta_pred = []
                    for t_im in tta_list:
                        t_im = t_im.cuda()
                        binary_logits, no_empty_logits, t_logits = self.forward(t_im, z)
                        pred = torch.sigmoid(t_logits)
                        pred = unpad_im(pred.cpu().numpy())
                        tta_pred.append(pred)
                    batch_pred.extend(tta_transform(tta_pred, mode='out'))
                    
                # predict original batch
                binary_logits, no_empty_logits, final_logits = self.forward(im, z)
                pred = torch.sigmoid(final_logits)
                pred = unpad_im(pred.cpu().numpy())
                batch_pred.append(pred)
                # Average batch results
                batch_pred = np.mean(batch_pred, 0)
                # convert batch_pred to a binary image(0, 1)
                if threshold > 0:
                    batch_pred = batch_pred > threshold
                if return_rle:
                    batch_pred = batch_encode(batch_pred)
                
                if not i:
                    out = batch_pred
                    ids = index
                else:
                    out = np.concatenate([out, batch_pred], axis=0)
                    ids = np.concatenate([ids, index], axis=0)
                    
        if return_rle:
            out = dict(id=ids, rle_mask=out)
            out = pd.DataFrame(out)
        else:
            out = dict(id=ids, pred=out)
        return out
    
    def define_criterion(self, name):
        if name.lower == 'lovasz':
            self.criterion = Loss.Lovasz_Hinge(per_image=True)
            
    def save_best_model(self, metric):
        if (self.val_mode == 'max' and metric > self.best_metric) or (self.val_mode == 'min' and metric < self.best_metric):
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            if self.fold is not None:
                self.best_model_path = os.path.join(self.save_dir, 'Fold{:}_Epoch{}_Val{:.3f}'.format(self.fold, self.epoch, 
                                                                                                      metric))
            else:
                self.best_model_path = os.path.join(self.save_dir, 'Epoch{}_Val{:.3f}'.format(self.epoch, metric))
            
            torch.save(self.state_dict(), self.best_model_path)
     
    def save_training_log(self):
        d = dict()
        for tk, vk in zip(self.train_log.keys(), self.val_log.keys()):
            d['train_{}'.format(tk)] = self.train_log[tk]
            d['val_{}'.format(vk)] = self.val_log[vk]
            
        df = pd.DataFrame(d)
        df.index += 1
        df.index.name = 'Epoch'
        
        if self.fold is not None:
            p = os.path.join(self.save_dir, 'Fold{:}_TrainLog.csv'.format(self.fold))
        else:
            p = os.path.join(self.save_dir, 'TrainLog.csv')
        df.to_csv(p, sep=';')
        
        with open(p, 'a') as fd:
            fd.write(self.comment)
            
    def load_model(self, path=None, best_model=False):
        if best_model:
            self.load_state_dict(torch.load(self.best_model_path))
        else:
            self.load_state_dict(torch.load(path))
            
    def create_save_folder(self):
        name = type(self).__name__
        self.save_dir = os.path.join('./Saves', name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def plot_training_curve(self, show=True):
        fig, axs = plt.subplots(1, 3)
        for i, metric in enumerate(['loss', 'iou', 'mAP']):
            axs[i].plot(self.train_log[metric], 'ro-', label='Train')
            axs[i].plot(self.val_log[metric], 'bo-', label='Validation')
            axs[i].legend()
            axs[i].set_title(metric)
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel(metric)
        if show:
            plt.show()
            
        
            
            
       
        
    
                    
                                       
                    
                    
                
                
            
                
                
                
                
                
                
                
                
        
        
        
            
            
            
                  
            
                    
                         
        
                
            
        
