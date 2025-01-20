# -*- coding: utf-8 -*-
"""
Created on 2025-01-20 (Mon) 13:31:30

TODO
- Generalize the batch normalization function
- Add docstring to each function

@author: I.Azuma
"""
import copy
import itertools
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import rankdata
from tqdm import tqdm
from combat.pycombat import pycombat
import matplotlib.pyplot as plt

class ReferencePreprocessing():
    def __init__(self, verbose=True):
        self.bulk_df = None
        self.ref_df = None
        self.verbose = verbose

    def set_data(self, bulk_df, ref_df):
        """
        set input data
        ----------
        bulk_df : DataFrame
            deconvolution target data (for which we want to know the population of each immune cell). 
            Set the genes and samples in rows and columns, respectively.
            e.g. /data/input/mix_processed.csv
                                AN_1       AN_2       AN_3  ...     GAL_4     GAL_7     GAL_8
            0610005C13Rik  11.875867  12.148424  11.252824  ...  8.431683  7.700966  8.962926
            0610009B22Rik   8.734242   8.457720   8.184438  ...  8.663929  8.546301  8.989601
            0610010F05Rik   6.490900   6.421802   5.806085  ...  6.207227  6.382519  6.357380
            0610010K14Rik   8.564556   8.673014   8.348932  ...  8.512630  8.407845  8.454941
            0610012G03Rik   8.097187   7.933714   7.698964  ...  8.108420  7.913200  8.247160
            
        ref_df : DataFrame
            Reference data for estimate the population of the immune cells
            e.g. /data/input/ref_13types.csv
                           NK_GSE114827_1  NK_GSE114827_2  ...  NK_GSE103901_1  NK_GSE103901_2                                        ...                                
            0610005C13Rik       11.216826       23.474148  ...       15.364877       31.139537
            0610006L08Rik        0.000000        0.000000  ...        0.000000        0.000000
            0610009B22Rik     3801.734228     3838.271735  ...     1673.625997     1265.782913
            0610009E02Rik       38.492335       26.822363  ...       31.225440        5.380941
            0610009L18Rik      260.189214      144.910984  ...      639.627003      664.538873
            
        """
        self.bulk_df = bulk_df
        self.ref_df = ref_df
    
    def pre_processing(self, do_ann=True, ann_df=None, do_log2=True, do_quantile=True, do_trimming=True, do_drop=True):
        if do_ann:
            self.ref_df = annotation(self.ref_df, ann_df)
        
        if do_log2:
            self.ref_df = log2(copy.deepcopy(self.ref_df))
        
        if do_quantile:
            self.ref_df = quantile(copy.deepcopy(self.ref_df))
        
        if do_trimming:
            df_c = copy.deepcopy(self.ref_df)
            raw_batch = [t.split("_")[0] for t in df_c.columns.tolist()]
            batch_list = pd.Series(raw_batch).astype("category").cat.codes.tolist()
            self.ref_df = array_imputer(df_c,lst_batch=batch_list,trim=1.0,batch=True,trim_red=False,threshold=0.9,trategy="median")
        
        if do_drop:
            df_c = copy.deepcopy(self.ref_df)
            drop = df_c.replace(0,np.nan).dropna(how="all").fillna(0)
            self.ref_df = drop
    
    def narrow_intersec(self):
        """        
        Note that mix_data is already processed (trimmed) in general (log2 --> trim+impute --> batch norm --> QN).
        This is because the robustness of the analysis is reduced if the number of genes to be analyzed is not narrowed down to a certain extent.
        """   
        # trimming
        bulk_data = copy.deepcopy(self.bulk_df)
        reference_data = copy.deepcopy(self.ref_df)
        
        self.bulk_df, self.ref_df = self.__intersection_index(bulk_data,reference_data) # update
        print("narrowd gene number :",len(self.ref_df))
    
    def __intersection_index(self,df,df2):
        ind1 = df.index.tolist()
        ind2 = df2.index.tolist()
        df.index = [i.upper() for i in ind1]
        df2.index = [i.upper() for i in ind2]
        ind = list(set(df.index) & set(df2.index))
        df = df.loc[ind,:]
        df2 = df2.loc[ind,:]
        return df, df2

    def create_ref(self,do_plot=False,**kwargs):
        """
        create reference dataframe which contains signatures for each cell
        """
        if kwargs["log2"]:
            df2 = copy.deepcopy(self.ref_df)
            df2 = np.log2(df2+1)
            self.ref_df = df2

        # DEG extraction
        sep = kwargs["sep"]
        self.deg_extraction(sep_ind=sep,number=kwargs["number"],limit_CV=kwargs["limit_CV"],limit_FC=kwargs["limit_FC"])
        
        signature = self.get_genes() # union of each reference cell's signatures
        print("signature genes :",len(signature))
        ref_inter_df = copy.deepcopy(self.ref_df)
        self.sig_ref = ref_inter_df.loc[signature]
        if sep == "":
            final_ref = self.sig_ref
        else:
            final_ref = self.__df_median(self.sig_ref,sep=sep)
        if do_plot:
            sns.clustermap(final_ref,col_cluster=False,z_score=0,figsize=(6, 6))
            plt.show()

        s_gene = sorted(final_ref.index.tolist())
        self.final_ref = final_ref.loc[s_gene]
        self.final_bulk = self.bulk_df.loc[s_gene]
        
    def deg_extraction(self,sep_ind="_",number=150,limit_CV=0.1,limit_FC=1.5):
        """
        Define DEGs between the target and other one for the cells that make up the REFERENCE.
        e.g. B cell vs CD4, B cell vs CD8, ...

        Parameters
        ----------
        sep_ind : str
            Assume the situation that the columns name is like "CellName_GSEXXX_n". The default is "_".
        number : int
            Number of top genes considered as DEGs. The default is 150.
        limit_CV : float
            Coefficient of Variation threshold. The default is 0.3.
        limit_FC : TYPE, float
            Minimum threshold for logFC detection. The default is 1.5.

        """
        df_c = copy.deepcopy(self.ref_df)
        #cluster, self.samples = self.sepmaker(df=df_c,delimiter=sep_ind)
        #print(cluster)
        if sep_ind=="":
            immunes = df_c.columns.tolist()
        else:
            immunes = [t.split(sep_ind)[0] for t in df_c.columns.tolist()]
        df_c.columns = immunes
        self.min_FC = pd.DataFrame()
        self.pickup_genes_list = []
        self.__pickup_genes = []
        for c in sorted(list(set(immunes))):
            if sep_ind=="":
                self.df_target = pd.DataFrame(df_c[[c]])  # FIXME: 240813 
            else:
                self.df_target = pd.DataFrame(df_c[c]) # FIXME: 230117 
            self.tmp_summary = pd.DataFrame()
            for o in sorted(list(set(immunes))):
                if o == c:
                    pass
                else:
                    if sep_ind =="":
                        self.df_else = df_c[[o]]
                    else:
                        self.df_else = df_c[o]
                    self.__logFC()
                    df_logFC = self.df_logFC
                    df_logFC.columns = [o]
                    self.tmp_summary = pd.concat([self.tmp_summary,df_logFC],axis=1)
            tmp_min = self.tmp_summary.T.min()
            self.df_minFC = pd.DataFrame(tmp_min)
            self.__calc_CV()
            
            pickup_genes = self.__selection(number=number,limit_CV=limit_CV,limit_FC=limit_FC)
            self.pickup_genes_list.append(pickup_genes)
            self.min_FC = pd.concat([self.min_FC,tmp_min],axis=1)
        self.min_FC.columns = sorted(list(set(immunes)))
        self.pickup_genes_df=pd.DataFrame(self.pickup_genes_list).T.dropna(how="all")
        self.pickup_genes_df.columns = sorted(list(set(immunes)))
        curate = [[i for i in t if str(i)!='nan'] for t in self.pickup_genes_list]
        self.deg_dic = dict(zip(sorted(list(set(immunes))),curate))
    
    def __df_median(self,df,sep="_"):
        df_c = copy.deepcopy(df)
        df_c.columns=[i.split(sep)[0] for i in list(df_c.columns)]
        df_c = df_c.groupby(level=0,axis=1).median()
        return df_c
    
    ### calculation ###
    def __logFC(self):
        # calculate df_target / df_else logFC
        df_logFC = self.df_target.T.median() - self.df_else.T.median()
        df_logFC = pd.DataFrame(df_logFC)
        self.df_logFC = df_logFC
    
    def __calc_CV(self):
        """
        CV : coefficient of variation
        """
        df_CV = np.std(self.df_target.T) / np.mean(self.df_target.T)
        df_CV.index = self.df_target.index
        df_CV = df_CV.replace(np.inf,np.nan)
        df_CV = df_CV.replace(-np.inf,np.nan)
        df_CV = df_CV.dropna()
        self.df_CV=pd.DataFrame(df_CV)
    
    def __selection(self,number=50,limit_CV=0.1,limit_FC=1.5):
        self.__intersection()
        df_minFC=self.df_minFC
        df_CV=self.df_CV
        df_minFC=df_minFC.sort_values(0,ascending=False)
        genes=df_minFC.index.tolist()
    
        pickup_genes=[]
        ap = pickup_genes.append
        i=0
        while len(pickup_genes)<number:
            if len(genes)<i+1:
                pickup_genes = pickup_genes+[np.nan]*number
                if self.verbose:
                    print('not enough genes picked up')
            elif df_CV.iloc[i,0] < limit_CV and df_minFC.iloc[i,0] > limit_FC:
                ap(genes[i])
            i+=1
        else:
            self.__pickup_genes = self.__pickup_genes + pickup_genes
            return pickup_genes
    
    def __intersection(self):
        lis1 = list(self.df_minFC.index)
        lis2 = list(self.df_CV.index)
        self.df_minFC = self.df_minFC.loc[list(set(lis1)&set(lis2))]
        self.df_CV = self.df_CV.loc[list(set(lis1)&set(lis2))]
        
    def get_genes(self):
        self.__pickup_genes=[i for i in self.__pickup_genes if str(i)!='nan']
        self.__pickup_genes=list(set(self.__pickup_genes))
        return self.__pickup_genes


def annotation(df,ref_df, places:list=[0, 1]):
    """
    annotate row IDs to gene names
    Parameters
    ----------
    df : a dataframe to be analyzed
    ref_df : two rows of dataframe. e.g. ["Gene stable ID","MGI symbol"]
    places : list of positions of target rows in the ref_df
    """
    ref_df_dropna = ref_df.iloc[:,places].dropna(how='any', axis=0)
    id_lst = ref_df_dropna.iloc[:,0].tolist()
    symbol_lst = ref_df_dropna.iloc[:,1].tolist()
    conv_dict = dict(list(zip(id_lst, symbol_lst)))
    id_lst_raw = [str(x).split(".")[0] for x in df.index.tolist()] # ENSMUSG00000000049.12 --> ENSMUSG00000000049
    symbol_lst_new = [conv_dict.get(x, np.nan) for x in id_lst_raw]
    df_conv = copy.deepcopy(df)
    df_conv["symbol"] = symbol_lst_new # add new col
    df_conv = df_conv.dropna(subset=["symbol"])
    df_conv = df_conv.groupby("symbol").median() # take median value for duplication rows
    return df_conv

def array_imputer(df,threshold=0.9,strategy="median",trim=1.0,batch=False,lst_batch=[], trim_red=True):
    """
    imputing nan and trim the values less than 1
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed
    
    threshold: float, default 0.9
        determine whether imupting is done or not dependent on ratio of not nan
        
    strategy: str, default median
        indicates which statistics is used for imputation
        candidates: "median", "most_frequent", "mean"
    
    lst_batch : lst, int
        indicates batch like : 0, 0, 1, 1, 1, 2, 2
    
    Returns
    ----------
    res: a dataframe
    
    """
    df_c = copy.deepcopy(df)
    if (type(trim)==float) or (type(trim)==int):
        df_c = df_c.where(df_c > trim)
    else:
        pass
    df_c = df_c.replace(0,np.nan)
    if batch:
        lst = []
        ap = lst.append
        for b in range(max(lst_batch)+1):
            place = [i for i, x in enumerate(lst_batch) if x == b]
            print("{0} ({1} sample)".format(b,len(place)))
            temp = df_c.iloc[:,place]
            if temp.shape[1]==1:
                ap(pd.DataFrame(temp))
            else:
                thresh = int(threshold*float(len(list(temp.columns))))
                temp = temp.dropna(thresh=thresh)
                imr = SimpleImputer(strategy=strategy)
                imputed = imr.fit_transform(temp.values.T) # impute in columns
                ap(pd.DataFrame(imputed.T,index=temp.index,columns=temp.columns))
        if trim_red:
            df_res = pd.concat(lst,axis=1)
            df_res = df_res.replace(np.nan,0) + 1
            print("redundancy trimming")
        else:
            df_res = pd.concat(lst,axis=1,join="inner")
    else:            
        thresh = int(threshold*float(len(list(df_c.columns))))
        df_c = df_c.dropna(thresh=thresh)
        imr = SimpleImputer(strategy=strategy)
        imputed = imr.fit_transform(df_c.values.T) # impute in columns
        df_res = pd.DataFrame(imputed.T,index=df_c.index,columns=df_c.columns)
    return df_res


def trimming(df, log=True, trimming=True, batch=False, lst_batch=[], trim_red=False, threshold=0.9):
    df_c = copy.deepcopy(df)
    # same index median
    df_c.index = [str(i) for i in df_c.index]
    df2 = pd.DataFrame()
    dup = df_c.index[df_c.index.duplicated(keep="first")]
    gene_list = pd.Series(dup).unique().tolist()
    if len(gene_list) != 0:
        for gene in gene_list:
            new = df_c.loc[:,gene].median()
            df2.loc[gene] = new
        df_c = df_c.drop(gene_list)
        df_c = pd.concat([df_c,df2.T])
    
    if trimming:
        if len(df_c.T) != 1:    
            df_c = array_imputer(df_c,lst_batch=lst_batch,batch=batch,trim_red=trim_red,threshold=threshold)
        else:
            df_c = df_c.where(df_c>1)
            df_c = df_c.dropna()
    else:
        df_c = df_c.dropna()

    # log conversion
    if log:
        df_c = df_c.where(df_c>=0)
        df_c = df_c.dropna()
        df_c = np.log2(df_c+1)
    else:
        pass
    return df_c

def batch_norm(df,lst_batch=[]):
    """
    batch normalization with combat
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed
    
    lst_batch : lst, int
        indicates batch like : 0, 0, 1, 1, 1, 2, 2
    
    """
    comb_df = pycombat(df,lst_batch)
    return comb_df

def multi_batch_norm(df,lst_lst_batch=[[],[]],do_plots=True):
    """
    batch normalization with combat for loop
    
    Note that the order of normalization is important. Begin with the broadest batch and move on to more specific batches of corrections.
    
    e.g. sex --> area --> country
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed
    
    lst_batch : lst, int
        indicates batch like : [[0,0,1,1,1,1],[0,0,1,1,2,2]]
    
    """
    df_c = df.copy() # deep copy
    for lst_batch in tqdm(lst_lst_batch):
        comb = batch_norm(df_c,lst_batch)
        df_c = comb # update
        if do_plots:
            for i in range(5):
                plt.hist(df_c.iloc[:,i],bins=200,alpha=0.8)
            plt.show()
        else:
            pass
    return df_c

def quantile(df,method="median"):
    """
    quantile normalization of dataframe (variable x sample)
    
    Parameters
    ----------
    df: dataframe
        a dataframe subjected to QN
    
    method: str, default "median"
        determine median or mean values are employed as the template    

    """
    #print("quantile normalization (QN)")
    df_c = df.copy() # deep copy
    lst_index = list(df_c.index)
    lst_col = list(df_c.columns)
    n_ind = len(lst_index)
    n_col = len(lst_col)

    ### prepare mean/median distribution
    x_sorted = np.sort(df_c.values,axis=0)[::-1]
    if method=="median":
        temp = np.median(x_sorted,axis=1)
    else:
        temp = np.mean(x_sorted,axis=1)
    temp_sorted = np.sort(temp)[::-1]

    ### prepare reference rank list
    x_rank_T = np.array([rankdata(v,method="ordinal") for v in df_c.T.values])

    ### conversion
    rank = sorted([v + 1 for v in range(n_ind)],reverse=True)
    converter = dict(list(zip(rank,temp_sorted)))
    converted = []
    converted_ap = converted.append  
    for i in range(n_col):
        transient = [converter[v] for v in list(x_rank_T[i])]
        converted_ap(transient)

    np_data = np.matrix(converted).T
    df2 = pd.DataFrame(np_data)
    df2.index = lst_index
    df2.columns = lst_col
    return df2

def log2(df):
    f_add = lambda x: x+1
    log_df = df.apply(f_add)
    log_df = np.log2(log_df)
    return log_df

def low_cut(df,threshold=1.0):
    df_c = copy.deepcopy(df)
    if (type(threshold)==float) or (type(threshold)==int):
        cut_df = df_c.where(df_c > threshold)
    else:
        pass
    return cut_df

def standardz_sample(x):
    pop_mean = x.mean(axis=0)
    pop_std = x.std(axis=0)+ np.spacing(1) # np.spacing(1) == np.finfo(np.float64).eps
    df = (x - pop_mean).divide(pop_std)
    df = df.replace(np.inf,np.nan)
    df = df.replace(-np.inf,np.nan)
    df = df.dropna()
    print('standardz population control')
    return df

def ctrl_norm(df,ctrl="C"):
    """normalization with ctrl samples"""
    ctrl_samples = []
    for t in df.index.tolist():
        if t.split("_")[0]==ctrl:
            ctrl_samples.append(t)
    ctrl_df = df.loc[ctrl_samples]
    
    ctrl_mean = ctrl_df.mean() # mean value of ctrl
    ctrl_std = ctrl_df.std() # std of ctrl
    
    norm_df = (df-ctrl_mean)/ctrl_std
    return norm_df

def drop_all_missing(df):
    replace = df.replace(0,np.nan)
    drop = replace.dropna(how="all") # remove rows whose all values are missing
    res = drop.fillna(0)
    print(len(df)-len(res),"rows are removed")
    return res

def freq_norm(df,marker_dic,ignore_others=True):
    """
    Normalize by sum of exression
    ----------
    df : DataFrame
        Genes in row and samples in column.
             PBMCs, 17-002  PBMCs, 17-006  ...  PBMCs, 17-060  PBMCs, 17-061
    AIF1          9.388634       8.354677  ...       8.848500       9.149019
    AIM2          4.675251       4.630904  ...       4.830909       4.831925
    ALOX5AP       9.064822       8.891569  ...       9.420134       9.192017
    APBA2         4.313265       4.455105  ...       4.309868       4.338142
    APEX1         7.581810       7.994079  ...       7.604995       7.706539
                   ...            ...  ...            ...            ...
    VCAN          8.213386       7.018457  ...       9.050750       8.263430
    VIPR1         6.436875       6.281543  ...       5.973437       6.622016
    ZBTB16        4.687727       4.618193  ...       4.730128       4.546280
    ZFP36        12.016052      11.514114  ...      11.538242      12.271717
    ZNF101        5.288079       5.250802  ...       5.029970       5.141903
    
    marker_dic : dict

    """
    others = sorted(list(set(df.index.tolist()) - set(itertools.chain.from_iterable(marker_dic.values()))))
    if len(others)>0:
        other_dic = {'others':others}
        #marker_dic = marker_dic | other_dic # Python 3.9
        marker_dic = {**marker_dic,**other_dic}

    # normalize
    use_k = []
    use_v = []
    for i,k in enumerate(marker_dic):
        if len(marker_dic.get(k))>0:
            use_k.append(k)
            use_v.append(marker_dic.get(k))
        else:
            pass
    marker_dic = dict(zip(use_k,use_v))
    
    cell_sums = []
    for i,k in enumerate(marker_dic):
        if ignore_others:
            if k == 'others':
                cell_sums.append(-1)
            else:
                common_v = sorted(list(set(marker_dic.get(k)) & set(df.index.tolist())))
                tmp_df = df.loc[common_v] # expression of markers
                tmp_sum = tmp_df.T.sum() # sum of expression level
                cell_sum = sum(tmp_sum)
                cell_sums.append(cell_sum)
        else:
            common_v = sorted(list(set(marker_dic.get(k)) & set(df.index.tolist())))
            tmp_df = df.loc[common_v] # expression of markers
            tmp_sum = tmp_df.T.sum() # sum of expression level
            cell_sum = sum(tmp_sum)
            cell_sums.append(cell_sum)
    
    base = max(cell_sums) # unify to maximum value
    r = [base/t for t in cell_sums]
    
    norm_df = pd.DataFrame()
    for i,k in enumerate(marker_dic):
        common_v = sorted(list(set(marker_dic.get(k)) & set(df.index.tolist())))
        tmp_df = df.loc[common_v] # expression of markers
        if ignore_others:
            if k == 'others':
                tmp_norm = tmp_df
            else:
                tmp_norm = tmp_df*r[i]
        else:
            tmp_norm = tmp_df*r[i]
        norm_df = pd.concat([norm_df,tmp_norm])
    
    # for multiple marker origin
    sample_name = norm_df.columns.tolist()[0]
    sort_norm = norm_df.sort_values(sample_name,ascending=False)

    # Gene duplications caused by multiple corrections are averaged.
    sort_norm['gene_name'] = sort_norm.index.tolist()
    trim_df= sort_norm.groupby("gene_name").mean() 

    #trim_df = sort_norm[~sort_norm.index.duplicated(keep='first')] # pick up the 1st one.
    return trim_df

def size_norm(df,marker_dic):
    """
    Normalize by gene size (number).
    ----------
    df : DataFrame
        Genes in row and samples in column.
             PBMCs, 17-002  PBMCs, 17-006  ...  PBMCs, 17-060  PBMCs, 17-061
    AIF1          9.388634       8.354677  ...       8.848500       9.149019
    AIM2          4.675251       4.630904  ...       4.830909       4.831925
    ALOX5AP       9.064822       8.891569  ...       9.420134       9.192017
    APBA2         4.313265       4.455105  ...       4.309868       4.338142
    APEX1         7.581810       7.994079  ...       7.604995       7.706539
                   ...            ...  ...            ...            ...
    VCAN          8.213386       7.018457  ...       9.050750       8.263430
    VIPR1         6.436875       6.281543  ...       5.973437       6.622016
    ZBTB16        4.687727       4.618193  ...       4.730128       4.546280
    ZFP36        12.016052      11.514114  ...      11.538242      12.271717
    ZNF101        5.288079       5.250802  ...       5.029970       5.141903
    
    marker_dic : dict

    """
    max_size = max([len(t) for t in marker_dic.values()])
    norm_df = pd.DataFrame()
    for i,k in enumerate(marker_dic):
        common_v = sorted(list(set(marker_dic.get(k)) & set(df.index.tolist())))
        tmp_size = len(common_v)
        r = max_size / tmp_size
        tmp_df = df.loc[common_v] # expression of markers
        tmp_norm = tmp_df*r
        norm_df = pd.concat([norm_df,tmp_norm])
    return norm_df

def norm_total_res(total_res,base_names=['Monocytes', 'NK cells', 'B cells naive', 'B cells memory', 'T cells CD4 naive', 'T cells CD4 memory', 'T cells CD8', 'T cells gamma delta']):
    norm_total_res = []
    for tmp_df in total_res:
        tmp_df = tmp_df[base_names]
        tmp_sum = tmp_df.T.sum()
        r = 1/tmp_sum
        norm_res = (tmp_df.T*r).T
        norm_total_res.append(norm_res)
    return norm_total_res

def norm_val(val_df,base_names=['Naive B', 'Memory B', 'CD8 T', 'Naive CD4 T', 'Gamma delta T', 'NK', 'Monocytes']):
    tmp_df = val_df[base_names]
    tmp_sum = tmp_df.T.sum()
    r = 1/tmp_sum
    norm_res = (tmp_df.T*r).T
    return norm_res
