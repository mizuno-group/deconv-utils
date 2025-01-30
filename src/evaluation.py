# -*- coding: utf-8 -*-
"""
Created on 2025-01-20 (Mon) 13:29:57

Evaluation and visualization tools for deconvolution results.

@author: I.Azuma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from matplotlib import colors as mcolors
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DeconvPlot():
    def __init__(self,deconv_df,val_df,dec_name=['B cells naive'],val_name=['Naive B'],
                do_plot=True,figsize=(6,6),dpi=300,plot_size=100):
        self.deconv_df = deconv_df
        self.val_df = val_df
        self.dec_name = dec_name
        self.val_name = val_name
        self.do_plot = do_plot
        self.figsize = figsize
        self.dpi = dpi
        self.plot_size = plot_size

        self.xlabel = 'Estimated Proportion'
        self.ylabel = 'True Proportion'
        self.label_size = 20
        self.tick_size = 15
    
    def calc_metrics(self,x,y):
        # Pearson correlation and p-value
        r, pvalue = stats.pearsonr(x,y)
        r = round(r,4)
        if pvalue < 0.01:
            pvalue = '{:.2e}'.format(pvalue)
        else:
            pvalue = round(pvalue,3)
        
        # CCC (Concordance Correlation Coefficient)
        mean_y = np.mean(y)
        mean_x = np.mean(x)
        var_y = np.var(y)
        var_x = np.var(y)
        sd_y = np.std(y)
        sd_x = np.std(x)

        # Calculate CCC
        numerator = 2 * r * sd_y * sd_x
        denominator = var_y + var_x + (mean_y - mean_x) ** 2
        ccc = numerator / denominator
        
        # RMSE and MAE
        rmse = round(np.sqrt(mean_squared_error(x, y)),4)
        mae = round(mean_absolute_error(x, y),4)

        return {'R':r,'P':pvalue,'CCC':ccc,'RMSE':rmse,'MAE':mae}


    def plot_simple_corr(self,color='tab:blue',title='Naive B',target_samples=None):
        """
        Correlation Scatter Plotting
        Format of both input dataframe is as follows
        Note that the targe data contains single treatment group (e.g. APAP treatment only)
        
                    B       CD4       CD8      Monocytes        NK  Neutrophils
        Donor_1 -0.327957 -0.808524 -0.768420   0.311360  0.028878     0.133660
        Donor_2  0.038451 -0.880116 -0.278970  -1.039572  0.865344    -0.437588
        Donor_3 -0.650633  0.574758 -0.498567  -0.796406 -0.100941     0.035709
        Donor_4 -0.479019 -0.005198 -0.675028  -0.787741  0.343481    -0.062349
        
        """
        total_x = self.deconv_df[self.dec_name].sum(axis=1).tolist()
        total_y = self.val_df[self.val_name].sum(axis=1).tolist()
        
        performance = self.calc_metrics(total_x,total_y)
        total_cor = performance['R']
        pvalue = performance['P']
        rmse = performance['RMSE']
        ccc = performance['CCC']
        
        x_min = min(min(total_x),min(total_y))
        x_max = max(max(total_x),max(total_y))
        
        if self.do_plot:
            fig,ax = plt.subplots(figsize=self.figsize,dpi=self.dpi)
            if target_samples is None:
                plt.scatter(total_x,total_y,alpha=1.0,s=self.plot_size,c=color)
            else:
                markers1 = ["o", "^", "+", ",", "v",  "<", ">"]
                for mi,d in enumerate(target_samples):
                    tmp1 = self.deconv_df.filter(regex="^"+d+"_",axis=0)
                    tmp2 = self.val_df.filter(regex="^"+d+"_",axis=0)
                    res1 = tmp1[self.dec_name].sum(axis=1).tolist()
                    res2 = tmp2[self.val_name].sum(axis=1).tolist()
                    plt.scatter(res1,res2,alpha=1.0,s=self.plot_size,c=color,marker=markers1[mi])

            plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
            plt.text(1.0,0.20,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.15,'P = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.10,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes, fontsize=15)
            plt.text(1.0,0.05,'CCC = {}'.format(str(round(ccc,3))), transform=ax.transAxes, fontsize=15)
            
            plt.xlabel(self.xlabel,fontsize=self.label_size)
            plt.ylabel(self.ylabel,fontsize=self.label_size)
            xlocs, _ = plt.xticks()
            ylocs, _ = plt.yticks()
            tick_min = max(0.0,min(xlocs[0],ylocs[0]))
            tick_max = max(xlocs[-1],ylocs[-1])
            step = (tick_max-tick_min)/5
            plt.xticks(np.arange(tick_min,tick_max,step),fontsize=self.tick_size)
            plt.yticks(np.arange(tick_min,tick_max,step),fontsize=self.tick_size)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            ax.set_axisbelow(True)
            ax.grid(color="#ababab",linewidth=0.5)
            plt.title(title,fontsize=self.label_size)
            plt.show()
        else:
            pass

        return performance,total_x,total_y

    
    def overlap_singles(self,evalxy, title_list=['Naive B','Naive CD4 T','CD8 T','NK','Monocytes']):
        total_x = []
        for t in evalxy[0]:
            total_x.extend(t)
        total_y = []
        for t in evalxy[1]:
            total_y.extend(t)

        performance = self.calc_metrics(total_x,total_y)
        total_cor = performance['R']
        pvalue = performance['P']
        rmse = performance['RMSE']
        ccc = performance['CCC']

        x_min = min(min(total_x),min(total_y))
        x_max = max(max(total_x),max(total_y))

        fig,ax = plt.subplots(figsize=self.figsize,dpi=self.dpi)
        for idx in range(len(evalxy[0])):
            res1 = evalxy[0][idx]
            res2 = evalxy[1][idx]
            cell = title_list[idx]

            plt.scatter(res1,res2,alpha=0.8,s=60,label=cell)
            plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)

        plt.plot([x_min,x_max],[x_min,x_max],linewidth=2,color='black',linestyle='dashed',zorder=-1)
        plt.text(1.0,0.20,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes, fontsize=15)
        plt.text(1.0,0.15,'P = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=15)
        plt.text(1.0,0.10,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes, fontsize=15)
        plt.text(1.0,0.05,'CCC = {}'.format(str(round(ccc,3))), transform=ax.transAxes, fontsize=15)
        #plt.legend(shadow=True)
        plt.xlabel('Estimated Proportion',fontsize=12)
        plt.ylabel('True Proportion',fontsize=12)
        plt.xticks(fontsize=self.tick_size)
        plt.yticks(fontsize=self.tick_size)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        #plt.title(title,fontsize=12)
        plt.legend(shadow=True,bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()

def eval_deconv(dec_name_list=[[0],[1]], val_name_list=[["Monocytes"],["CD4Tcells"]], 
                deconv_df=None, y_df=None):
    # overall
    assert len(dec_name_list) == len(val_name_list)
    tab_colors = mcolors.TABLEAU_COLORS
    color_list = list(tab_colors.keys())
    loop_n = len(dec_name_list) // len(color_list)
    color_list = color_list * (loop_n+1)
    overall_res = []
    for i in range(len(dec_name_list)):
        dec_name = dec_name_list[i]
        val_name = val_name_list[i]
        plot_dat = DeconvPlot(deconv_df=deconv_df,val_df=y_df,dec_name=dec_name,val_name=val_name,plot_size=20,dpi=50)
        res = plot_dat.plot_simple_corr(color=color_list[i],title=f'Topic:{dec_name} vs {val_name}',target_samples=None)
        overall_res.append(res)
    
    return overall_res


def main():
    WORKSPACE = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv/github/deconv-utils'
    os.chdir(WORKSPACE)
    # load deconvolution result adn ground truth
    deconv_df = pd.read_csv('./data/GSE65133/CIBERSORT_GSE65133_LM22_Results.csv',index_col=0)
    val_df = pd.read_csv('./data/GSE65133/ground_truth.csv',index_col=0)

    # scaling
    deconv_df = deconv_df.div(deconv_df.sum(axis=1),axis=0)
    val_df = val_df.div(val_df.sum(axis=1),axis=0)

    dp = DeconvPlot(deconv_df=deconv_df,val_df=val_df,dec_name=['B cells naive'],val_name=['Naive B'])
    dp.plot_simple_corr()


if __name__ == '__main__':
    main()
