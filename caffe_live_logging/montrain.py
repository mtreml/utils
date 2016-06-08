# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:03:39 2016

@author: atreo
"""

import matplotlib.pylab as plt
import numpy as np
import os

class BiasFig():
    """Creates a figure with average absolute bias of each layer in the net
        in one axis from a dataframe and saves it at every update
    
       Updates dynamically with BiasFig.update(net, dataframe)
    """
    def __init__(self, net, folder, **kwargs):
    
        if not os.path.isdir(os.path.join(folder, 'plots')):
            os.mkdir(os.path.join(folder, 'plots'))
        plt.ioff()
        self.bias_fig = plt.figure('Bias')  
        self.ax1 = self.bias_fig.add_subplot(111)
        self.plots = {}
        i = 0        
        for p in net.params:
            self.plots['p'+str(i)] = self.ax1.plot([], [], '-', label=p+', <|b|>')[0] # MIND: needs unpacking
            i += 1
        plt.xlabel('iterations')
        self.handles1, self.labels1 = self.ax1.get_legend_handles_labels()
        self.lgd1 = self.ax1.legend(self.handles1, self.labels1, loc='upper center', bbox_to_anchor=(0.5,-0.2))   
        self.ax1.grid(True)
        plt.draw()  
        
    
    def update(self, net, df, folder):

        i = 0   
        for p in net.params:

            self.plots['p'+str(i)].set_xdata(np.append(self.plots['p'+str(i)].get_xdata(), df.loc[len(df)-1]['']['NumIters']))# MIND: Use [''] for NumIters due to multi-index dataframe
            self.plots['p'+str(i)].set_ydata(np.append(self.plots['p'+str(i)].get_ydata(), df.loc[len(df)-1][p]['bias']))

            i += 1            
            
        self.ax1.relim()
        self.ax1.autoscale_view()
        plt.draw() 
    
        self.bias_fig.savefig(os.path.join(folder, os.path.join('plots', 'Bias.png')), bbox_extra_artists=(self.lgd1,), bbox_inches="tight")       



class WeightsFig():
    """Creates a figure with average absolute weights of each layer in the net
        in one axis from a dataframe and saves it at every update
    
       Updates dynamically with WeightsFig.update(net, dataframe)
    """
    def __init__(self, net, folder, **kwargs):
    
        if not os.path.isdir(os.path.join(folder, 'plots')):
            os.mkdir(os.path.join(folder, 'plots'))
        plt.ioff()
        self.weights_fig = plt.figure('Weights')  
        self.ax1 = self.weights_fig.add_subplot(111)
        self.plots = {}
        i = 0        
        for p in net.params:
            self.plots['p'+str(i)] = self.ax1.plot([], [], '-', label=p+', <|W|>')[0] # MIND: needs unpacking
            i += 1
        plt.xlabel('iterations')
        self.handles1, self.labels1 = self.ax1.get_legend_handles_labels()
        self.lgd1 = self.ax1.legend(self.handles1, self.labels1, loc='upper center', bbox_to_anchor=(0.5,-0.2))   
        self.ax1.grid(True)
        plt.draw()  
        
    
    def update(self, net, df, folder):
        
        i = 0   
        for p in net.params:

            self.plots['p'+str(i)].set_xdata(np.append(self.plots['p'+str(i)].get_xdata(), df.loc[len(df)-1]['']['NumIters']))# MIND: Use [''] for NumIters due to multi-index dataframe
            self.plots['p'+str(i)].set_ydata(np.append(self.plots['p'+str(i)].get_ydata(), df.loc[len(df)-1][p]['weight']))
         
            i += 1            
            
        self.ax1.relim()
        self.ax1.autoscale_view()
        plt.draw() 
    
        self.weights_fig.savefig(os.path.join(os.path.join(folder, 'plots'), 'Weights.png'), bbox_extra_artists=(self.lgd1,), bbox_inches="tight")       
            



class LossFig():   
    """Creates a figure with loss and average loss and iteration number
        in one axis from a dataframe and saves it at every update
       
       Can be used to plot Test- & Train-loss individually
    
       Updates dynamically with LossFig.update(dataframe)
    """
    def __init__(self, folder, title='', **kwargs):
    
        if not os.path.isdir(os.path.join(folder, 'plots')):
            os.mkdir(os.path.join(folder, 'plots'))
        plt.ioff()
        self.title = title
        self.loss_fig = plt.figure(self.title + 'Loss')  
        self.ax1 = self.loss_fig.add_subplot(111)
        self.p0, = self.ax1.plot([], [], 'k-', label=title+'-loss')
        self.p1, = self.ax1.plot([], [], 'b-', label=title+'-av_loss')
        plt.xlabel('iterations')
        self.handles1, self.labels1 = self.ax1.get_legend_handles_labels()
        self.lgd1 = self.ax1.legend(self.handles1, self.labels1, loc='upper center', bbox_to_anchor=(0.5,-0.2))   
        self.ax1.grid(True)
        plt.draw()


    def update(self, df, folder):

        self.p0.set_xdata(np.append(self.p0.get_xdata(), df.loc[len(df)-1]['NumIters']))
        self.p0.set_ydata(np.append(self.p0.get_ydata(), df.loc[len(df)-1]['loss']))
        self.p1.set_xdata(np.append(self.p1.get_xdata(), df.loc[len(df)-1]['NumIters']))
        self.p1.set_ydata(np.append(self.p1.get_ydata(), df.loc[len(df)-1]['av_loss']))
            
        self.ax1.relim()
        self.ax1.autoscale_view()
        plt.draw() 
    
        self.loss_fig.savefig(os.path.join(os.path.join(folder, 'plots'), self.title + 'Loss.png'), bbox_extra_artists=(self.lgd1,), bbox_inches="tight")       
            

class CompareLossFig():   
    """Creates a figure with comparison of average loss (Test & Train) and iteration number
        in one axis from a dataframe and saves it at every update
    
       Updates dynamically with CompareLossFig.update(train_df, test_df)
    """
    def __init__(self, folder, **kwargs):
    
        if not os.path.isdir(os.path.join(folder, 'plots')):
            os.mkdir(os.path.join(folder, 'plots'))
        plt.ioff()
        self.loss_fig = plt.figure('Loss')  
        self.ax1 = self.loss_fig.add_subplot(111)
        self.p0, = self.ax1.plot([], [], 'm-', label='Train loss')
        self.p1, = self.ax1.plot([], [], 'c-', label='Test loss')
        plt.xlabel('iterations')
        self.handles1, self.labels1 = self.ax1.get_legend_handles_labels()
        self.lgd1 = self.ax1.legend(self.handles1, self.labels1, loc='upper center', bbox_to_anchor=(0.5,-0.2))   
        self.ax1.grid(True)
        plt.draw()


    def update(self, train_df, test_df, folder):         

        self.p0.set_xdata(np.append(self.p0.get_xdata(), train_df.loc[len(train_df)-1]['NumIters']))
        self.p0.set_ydata(np.append(self.p0.get_ydata(), train_df.loc[len(train_df)-1]['av_loss']))

        self.p1.set_xdata(np.append(self.p1.get_xdata(), test_df.loc[len(train_df)-1]['NumIters']))
        self.p1.set_ydata(np.append(self.p1.get_ydata(), test_df.loc[len(train_df)-1]['av_loss']))
            
        self.ax1.relim()
        self.ax1.autoscale_view()
        plt.draw() 
    
        self.loss_fig.savefig(os.path.join(os.path.join(folder, 'plots'), 'Loss.png'), bbox_extra_artists=(self.lgd1,), bbox_inches="tight")       
               

class MetricsFig():   
    """Creates a figure with different metrics and iteration number
        in one axis from a dataframe and saves it at every update
        
       Can be used to plot Test- & Train-metrics individually
    
       Updates dynamically with MetricsLossFig.update(dataframe)
    """
    def __init__(self, folder, title='', **kwargs):  
        
        if not os.path.isdir(os.path.join(folder, 'plots')):
            os.mkdir(os.path.join(folder, 'plots'))
        plt.ioff()
        self.title = title    
        self.metrics_fig = plt.figure(self.title + 'Metrics')
        self.ax2 = self.metrics_fig.add_subplot(111)
        self.p1, = self.ax2.plot([], [], 'r-', label='Av Pixel accuracy')
        self.p2, = self.ax2.plot([], [], 'b-', label='Av Mean-Per-Class accuracy')
        self.p3, = self.ax2.plot([], [], 'g-', label='Av Mean-Per-Class IU')
        self.p4, = self.ax2.plot([], [], 'k-', label='Av Freq. weigh. mean IU')
        plt.xlabel('iterations')
        self.handles2, self.labels2 = self.ax2.get_legend_handles_labels()
        self.lgd2 = self.ax2.legend(self.handles2, self.labels2, loc='upper center', bbox_to_anchor=(0.5,-0.2))
        self.ax2.grid(True)    
        plt.draw()
            
            
    def update(self, df, folder):    

        self.p1.set_xdata(np.append(self.p1.get_xdata(), df.loc[len(df)-1]['NumIters']))
        self.p1.set_ydata(np.append(self.p1.get_ydata(), df.loc[len(df)-1]['av_pix_acc']))
        
        self.p2.set_xdata(np.append(self.p2.get_xdata(), df.loc[len(df)-1]['NumIters']))
        self.p2.set_ydata(np.append(self.p2.get_ydata(), df.loc[len(df)-1]['av_mean_acc']))
        
        self.p3.set_xdata(np.append(self.p3.get_xdata(), df.loc[len(df)-1]['NumIters']))
        self.p3.set_ydata(np.append(self.p3.get_ydata(), df.loc[len(df)-1]['av_iu']))
        
        self.p4.set_xdata(np.append(self.p4.get_xdata(), df.loc[len(df)-1]['NumIters']))
        self.p4.set_ydata(np.append(self.p4.get_ydata(), df.loc[len(df)-1]['av_fwavacc']))
            
        self.ax2.relim()
        self.ax2.autoscale_view()
        plt.draw()
        
        self.metrics_fig.savefig(os.path.join(os.path.join(folder, 'plots'), self.title + 'Metrics.png'), bbox_extra_artists=(self.lgd2,), bbox_inches="tight")


class CompareMetricsFig():   
    """Creates a figure with a comparison of metrics (Test & Train) and iteration number
        in one axis from a dataframe and saves it at every update
        
        MIGHT BE OUTDATED!
            
       Updates dynamically with CompareMetricsLossFig.update(train_df, test_df)
    """
    def __init__(self, folder, **kwargs):  
        
        if not os.path.isdir(os.path.join(folder, 'plots')):
            os.mkdir(os.path.join(folder, 'plots'))
        plt.ioff()
        self.metrics_fig = plt.figure('Metrics')
        self.ax2 = self.metrics_fig.add_subplot(111)

        self.p1, = self.ax2.plot([], [], 'ro-', label='TEST: Pixel accuracy')
        self.p5, = self.ax2.plot([], [], 'rv-', label='TRAIN: Pixel accuracy')
        
        self.p2, = self.ax2.plot([], [], 'bo-', label='TEST: Mean-Per-Class accuracy')
        self.p6, = self.ax2.plot([], [], 'bv-', label='TRAIN:Mean-Per-Class accuracy')
        
        self.p3, = self.ax2.plot([], [], 'go-', label='TEST: Mean-Per-Class IU')
        self.p7, = self.ax2.plot([], [], 'gv-', label='TRAIN:Mean-Per-Class IU')
        
        self.p4, = self.ax2.plot([], [], 'ko-', label='TEST: Freq. weigh. mean IU')
        self.p8, = self.ax2.plot([], [], 'kv-', label='TRAIN:Freq. weigh. mean IU')
        


        plt.xlabel('iterations')
        self.handles2, self.labels2 = self.ax2.get_legend_handles_labels()
        self.lgd2 = self.ax2.legend(self.handles2, self.labels2, loc='upper center', bbox_to_anchor=(0.5,-0.2))
        self.ax2.grid(True)    
        plt.draw()
            
            
    def update(self, train_dict_list, test_dict_list, folder):  
  
        if test_dict_list != []:

            self.p1.set_xdata(np.append(self.p1.get_xdata(), test_dict_list[-1]['NumIters']))
            self.p1.set_ydata(np.append(self.p1.get_ydata(), test_dict_list[-1]['PixelAccuracy']))
        
            self.p2.set_xdata(np.append(self.p2.get_xdata(), test_dict_list[-1]['NumIters']))
            self.p2.set_ydata(np.append(self.p2.get_ydata(), test_dict_list[-1]['MeanAccuracy']))
        
            self.p3.set_xdata(np.append(self.p3.get_xdata(), test_dict_list[-1]['NumIters']))
            self.p3.set_ydata(np.append(self.p3.get_ydata(), test_dict_list[-1]['IU']))
        
            self.p4.set_xdata(np.append(self.p4.get_xdata(), test_dict_list[-1]['NumIters']))
            self.p4.set_ydata(np.append(self.p4.get_ydata(), test_dict_list[-1]['FreqWeighMeanAcc']))

        if train_dict_list != []:

            self.p5.set_xdata(np.append(self.p5.get_xdata(), train_dict_list[-1]['NumIters']))
            self.p5.set_ydata(np.append(self.p5.get_ydata(), train_dict_list[-1]['PixelAccuracy']))
        
            self.p6.set_xdata(np.append(self.p6.get_xdata(), train_dict_list[-1]['NumIters']))
            self.p6.set_ydata(np.append(self.p6.get_ydata(), train_dict_list[-1]['MeanAccuracy']))
        
            self.p7.set_xdata(np.append(self.p7.get_xdata(), train_dict_list[-1]['NumIters']))
            self.p7.set_ydata(np.append(self.p7.get_ydata(), train_dict_list[-1]['IU']))
        
            self.p8.set_xdata(np.append(self.p8.get_xdata(), train_dict_list[-1]['NumIters']))
            self.p8.set_ydata(np.append(self.p8.get_ydata(), train_dict_list[-1]['FreqWeighMeanAcc']))
            
        self.ax2.relim()
        self.ax2.autoscale_view()
        plt.draw()
        
        self.metrics_fig.savefig(os.path.join(os.path.join(folder, 'plots'), 'Metrics.png'), bbox_extra_artists=(self.lgd2,), bbox_inches="tight")
            
            
