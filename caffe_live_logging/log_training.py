from __future__ import division
import numpy as np
import montrain
import pandas as pd
import diagnose_weights as dwi
import os
from threading import Thread
import queue


def observe_weights(solver, folder):
    """Invoke in solver.py
    """
     
    if 'weights_frame' not in globals():
        global weights_frame
        weights_frame = pd.DataFrame(columns=dwi.gen_pandas_struct(solver.net))
    # get or compute data
    current_df = dwi.get_mean_df(solver.net, solver.iter)
    weights_frame = pd.concat([weights_frame, current_df], ignore_index=True)   
    
    # save to csv    
    if not os.path.isdir(os.path.join(folder, 'csv')):
            os.mkdir(os.path.join(folder, 'csv'))
    weights_frame.to_csv(os.path.join(os.path.join(folder, 'csv'), 'weights_frame.csv'))

    # plot
    plot_weights(weights_frame, solver.net, folder)
    

def plot_weights(weights_frame, net, folder):
    """Plots the mean weights in all layers

    Initiates the plot-class if it does not exists
    Updates data live
    """
    
    # initiate plots            
    if 'weights_fig' not in globals():
        global weights_fig, bias_fig
        weights_fig = montrain.WeightsFig(net=net, folder=folder)
        bias_fig = montrain.BiasFig(net=net, folder=folder)
    weights_fig.update(net, weights_frame, folder)
    bias_fig.update(net, weights_frame, folder)
    

def observe_loss_and_acc(solver, folder):
    """Invoke in solver.py
    """
     
    if 'train_frame' not in globals():
        global train_frame, test_frame
        train_frame = pd.DataFrame(columns=('NumIters', 'loss', 'av_loss', 'pix_acc', 'av_pix_acc', 'mean_acc', 'av_mean_acc', 'iu', 'av_iu', 'fwavacc', 'av_fwavacc'))
        test_frame = pd.DataFrame(columns=('NumIters', 'loss', 'av_loss', 'pix_acc', 'av_pix_acc', 'mean_acc', 'av_mean_acc', 'iu', 'av_iu', 'fwavacc', 'av_fwavacc'))
    
    # get or compute data
    queue1 = queue.Queue()
    queue2 = queue.Queue()
    t1 = Thread(target=train_frame_update, args=(solver, train_frame, queue1))
    t2 = Thread(target=test_frame_update, args=(solver, test_frame, queue2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    train_frame = queue1.get()
    test_frame = queue2.get()
#    train_frame = train_frame_update(solver, train_frame)
#    test_frame = test_frame_update(solver, test_frame)
    
    # save to csv
    if not os.path.isdir(os.path.join(folder, 'csv')):
            os.mkdir(os.path.join(folder, 'csv'))
    train_frame.to_csv(os.path.join(os.path.join(folder, 'csv'), 'train_frame.csv'))
    test_frame.to_csv(os.path.join(os.path.join(folder, 'csv'), 'test_frame.csv'))
    
    # plot
    plot_diagnostics(train_frame, test_frame, folder)


def train_frame_update(solver, train_frame, queue):
    
    pix_acc, mean_acc, iu, fwavacc = compute_metrics(solver.net)
    loss = solver.net.blobs['loss'].data.flat[0]    
    train_frame.loc[len(train_frame)] = [solver.iter,
                                         loss,
                                         None,
                                         pix_acc,
                                         None,
                                         mean_acc,
                                         None,
                                         iu,
                                         None,
                                         fwavacc,
                                         None]
    # smoothen
    windowsize = 100
    train_frame.loc[len(train_frame)-1]['av_loss'] = np.mean(train_frame['loss'][-windowsize:-1])
    train_frame.loc[len(train_frame)-1]['av_pix_acc'] = np.mean(train_frame['pix_acc'][-windowsize:-1])
    train_frame.loc[len(train_frame)-1]['av_mean_acc'] = np.mean(train_frame['mean_acc'][-windowsize:-1])
    train_frame.loc[len(train_frame)-1]['av_iu'] = np.mean(train_frame['iu'][-windowsize:-1])
    train_frame.loc[len(train_frame)-1]['av_fwavacc'] = np.mean(train_frame['fwavacc'][-windowsize:-1])
    print('TRAIN LOSS:', loss)
    print('TRAIN AV LOSS:', train_frame.loc[len(train_frame)-1]['av_loss'])
    queue.put(train_frame)
#    return train_frame


def test_frame_update(solver, test_frame, queue):
    test_net = solver.test_nets[0]  
    test_net.share_with(solver.net)
    
    pix_acc, mean_acc, iu, fwavacc = compute_metrics(test_net)  
    test_net.forward()
    loss = test_net.blobs['loss'].data.flat[0]   
    test_frame.loc[len(test_frame)] = [solver.iter,
                                         loss,
                                         None,
                                         pix_acc,
                                         None,
                                         mean_acc,
                                         None,
                                         iu,
                                         None,
                                         fwavacc,
                                         None]
    # smoothen
    windowsize = 100
    test_frame.loc[len(test_frame)-1]['av_loss'] = np.mean(test_frame['loss'][-windowsize:-1])
    test_frame.loc[len(test_frame)-1]['av_pix_acc'] = np.mean(test_frame['pix_acc'][-windowsize:-1])
    test_frame.loc[len(test_frame)-1]['av_mean_acc'] = np.mean(test_frame['mean_acc'][-windowsize:-1])
    test_frame.loc[len(test_frame)-1]['av_iu'] = np.mean(test_frame['iu'][-windowsize:-1])
    test_frame.loc[len(test_frame)-1]['av_fwavacc'] = np.mean(test_frame['fwavacc'][-windowsize:-1])
#    print('TEST LOSS:', loss)
#    print('TEST AV LOSS:', test_frame.loc[len(test_frame)-1]['av_loss'])
    queue.put(test_frame)
#    return test_frame


def compute_metrics(net):

    n_cl = net.blobs['score'].channels
    hist = fast_hist(net.blobs['label'].data[0, 0].flatten(),
                                net.blobs['score'].data[0].argmax(0).flatten(),
                                n_cl)

    # mean loss
#    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'loss', loss)
    # overall accuracy
    pix_acc = np.diag(hist).sum() / hist.sum()
#    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'overall accuracy', pix_acc)
    # per-class accuracy
    mean_acc = np.nanmean(np.diag(hist) / hist.sum(1))
#    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'mean accuracy', mean_acc)
    # per-class IU
    iu_ = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    iu = np.nanmean(iu_)
#    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'mean IU', iu)
    # freq weighted acc    
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu_[freq > 0]).sum()
#    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'fwavacc', fwavacc)
    
    return pix_acc, mean_acc, iu, fwavacc


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)
   

def plot_diagnostics(train_frame, test_frame, folder):            
    """Plot different diagnostics whilst training

    Initiates the plot-class if it does not exists
    Updates data live
    """
    
    # initiate plots            
    if 'trainloss_fig' not in globals():
        global trainloss_fig, testloss_fig, compareloss_fig, trainmetrics_fig, testmetrics_fig
        trainloss_fig = montrain.LossFig(folder=folder, title='Train')
        testloss_fig = montrain.LossFig(folder=folder, title='Test')
        compareloss_fig = montrain.CompareLossFig(folder=folder)
        trainmetrics_fig = montrain.MetricsFig(folder=folder, title='Train')
        testmetrics_fig = montrain.MetricsFig(folder=folder, title='Test')
        

    trainloss_fig.update(train_frame, folder)
    testloss_fig.update(test_frame, folder)
    compareloss_fig.update(train_frame, test_frame, folder)
    trainmetrics_fig.update(train_frame, folder)
    testmetrics_fig.update(test_frame, folder)

