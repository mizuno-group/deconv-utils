# -*- coding: utf-8 -*-
"""
Created on 2025-01-20 (Mon) 13:54:45

TODO:
- Add docstrings

@author: I.Azuma
"""
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm

from sklearn.svm import NuSVR
from sklearn.linear_model import ElasticNet,Ridge,Lasso,HuberRegressor,LinearRegression

class RegressionDeconv():
    def __init__(self, bulk_df, ref_df):
        self.bulk_df = bulk_df
        self.ref_df = ref_df
    
    def deconv_ElasticNet(self, alpha=1, l1_ratio=0.05, max_iter=10000):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=1e-5, random_state=None, fit_intercept=True)
        model.fit(self.ref_df, self.bulk_df)
        res_df = pd.DataFrame(model.coef_,index=self.bulk_df.columns, columns=self.ref_df.columns)
        return res_df
    
    def deconv_Ridge(self, alpha=1, max_iter=10000):
        model = Ridge(alpha=alpha, max_iter=max_iter, tol=1e-5, random_state=None, fit_intercept=True)
        model.fit(self.ref_df, self.bulk_df)
        res_df = pd.DataFrame(model.coef_,index=self.bulk_df.columns, columns=self.ref_df.columns)
        return res_df
    
    def deconv_Lasso(self, alpha=1, max_iter=10000):
        model = Lasso(alpha=alpha, max_iter=max_iter, tol=1e-5, random_state=None, fit_intercept=True)
        model.fit(self.ref_df, self.bulk_df)
        res_df = pd.DataFrame(model.coef_,index=self.bulk_df.columns, columns=self.ref_df.columns)
        return res_df

    def deconv_NuSVR(self, nu=0.5,C=1.0,max_iter=10000):
        res_mat = []
        for i in range(len(self.bulk_df.columns)):
            model = NuSVR(nu=nu, C=C, kernel='linear', max_iter=max_iter)
            model.fit(self.ref_df, self.bulk_df.iloc[:,i])
            res_mat.append(model.coef_[0])
        res_df = pd.DataFrame(res_mat, index=self.bulk_df.columns, columns=self.ref_df.columns)
        return res_df
    
    def deconv_RLR(self, epsilon=1.35, max_iter=10000, alpha=0.0001):
        # RLR (Huber Robut Linear Regression)
        res_df = pd.DataFrame(index=self.bulk_df.columns, columns=self.ref_df.columns)
        for i in range(len(self.bulk_df.columns)):
            model = HuberRegressor(epsilon=epsilon, max_iter=max_iter, alpha=alpha, warm_start=False, fit_intercept=True)
            model.fit(self.ref_df, self.bulk_df.iloc[:,i])
            res_df.iloc[i,:] = model.coef_
        res_df = res_df.astype(float)
        return res_df
    
    def deconv_OLS(self):
        res_df = pd.DataFrame(index=self.bulk_df.columns, columns=self.ref_df.columns)
        for i in range(len(self.bulk_df.columns)):
            model = sm.OLS(np.array(self.bulk_df.iloc[:,i]),np.array(self.ref_df))
            result = model.fit()
            res_df.iloc[i,:] = result.params
        return res_df
    
    def deconv_NNLS(self):
        res_df = pd.DataFrame(index=self.bulk_df.columns, columns=self.ref_df.columns)
        for i in range(len(self.bulk_df.columns)):
            res_df.iloc[i,:] = sp.optimize.nnls(np.array(self.ref_df),np.array(self.bulk_df.iloc[:,i]))[0]
        return res_df
    
    def deconv_LR(self):
        res_df = pd.DataFrame(index=self.bulk_df.columns, columns=self.ref_df.columns)
        for i in range(len(self.bulk_df.columns)):
            model = LinearRegression()
            model.fit(self.ref_df, self.bulk_df.iloc[:,i])
            res_df.iloc[i,:] = model.coef_
        res_df = res_df.astype(float)
        return res_df


def main():
    WORKSPACE = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv/github/deconv-utils'
    os.chdir(WORKSPACE)
    from src import preprocessing as pp
    from src import evaluation

    # Load data and scaling
    raw_bulk = pd.read_csv("./data/GSE65133/GSE65133_expression.csv",index_col=0)
    raw_ref = pd.read_csv("./data/lm22_signature.csv",index_col=0)

    fxn = lambda x : np.log(x+1)
    log_bulk = raw_bulk.applymap(fxn)

    # Preprocessing of reference
    dat = pp.ReferencePreprocessing(verbose=True)
    dat.set_data(bulk_df=log_bulk, ref_df=raw_ref)
    dat.pre_processing(do_ann=False,ann_df=None,do_log2=True,do_quantile=False,do_trimming=False,do_drop=True)
    dat.narrow_intersec()
    dat.create_ref(sep="_",number=1000,limit_CV=10,limit_FC=1.5,log2=False,do_plot=False)

    ref_df = dat.final_ref
    bulk_df = dat.final_bulk
    
    # Run regression-based deconvolution
    rd = RegressionDeconv(bulk_df, ref_df)
    result_dict = {}
    result_dict["ElasticNet"] = rd.deconv_ElasticNet(alpha=1, l1_ratio=0.05, max_iter=10000)
    result_dict["Ridge"] = rd.deconv_Ridge(alpha=1, max_iter=10000)
    result_dict["Lasso"] = rd.deconv_Lasso(alpha=1, max_iter=10000)
    result_dict["NuSVR"] = rd.deconv_NuSVR(nu=0.5,C=1.0,max_iter=10000)
    result_dict["RLR"] = rd.deconv_RLR(epsilon=1.35, max_iter=10000, alpha=0.0001)
    result_dict["OLS"] = rd.deconv_OLS()
    result_dict["NNLS"] = rd.deconv_NNLS()
    result_dict["LR"] = rd.deconv_LR()

    # Evaluation
    deconv_df = result_dict["ElasticNet"].copy()
    deconv_df = deconv_df.div(deconv_df.sum(axis=1),axis=0)
    val_df = pd.read_csv('./data/GSE65133/ground_truth.csv',index_col=0)
    val_df = val_df.div(val_df.sum(axis=1),axis=0)

    ev = evaluation.DeconvPlot(deconv_df=result_dict["ElasticNet"],val_df=val_df,dec_name=['B cells naive'],val_name=['Naive B'])
    res_summary = ev.plot_simple_corr()


if __name__ == '__main__':
    main()

