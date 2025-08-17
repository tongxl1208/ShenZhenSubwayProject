import pandas as pd
import pingouin as pg
import numpy as np
from prophet import Prophet
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from scipy import stats
from scipy.stats import f_oneway, levene, bartlett, kruskal, shapiro, t


def caculate_yichang(df, last_interval,col_dict,taglist, shesd):    
    df_yichang = {'测量周期': [], '单次变形坐标': [], '检测点': [], 'P值':[], '异常值数量': []}
    resultdict = {}
    axis_dfs = []
    alpha = 0.025
    for name ,col in col_dict.items():
        count = 0
        for tag in taglist:
            R = df[(df['测点编号'] == tag)][col].to_numpy()
            mean_val = np.mean(R)
            res = shesd(R, period=None, max_anoms=None, alpha=alpha)
            R[res["anomaly_indices"]] = mean_val
            mask = (df['测点编号'] == tag)
            df.loc[mask, col] = R
            count += len(res["anomaly_indices"])                
            df_yichang['测量周期'].append(last_interval)
            df_yichang['单次变形坐标'].append(name)
            df_yichang['检测点'].append(tag)
            df_yichang['P值'].append(alpha)
            df_yichang['异常值数量'].append(len(res["anomaly_indices"]))
    resultdict['data'] = df_yichang
    resultdict['err'] = 0

    return resultdict, df
    

def caculate_zhengtai(df, col_dict, intervals, taglist):

    resultdict = {}

    shapiro_dict = {'测点编号':[], '测量周期': [],'单次变形坐标':[],'mean':[], 'std': [],'var': [], 
                'max': [], 'Q3': [], 'Q2':[], 'Q1':[], 'min': [], 'p值':[], 'Shapiro-Wilk':[], 'PP图':[]}
    
    for name ,col in col_dict.items():
        for interval in intervals:
            for tag in taglist:
                pp_dict = {}
                # 去掉绝对值大于1mm的值
                data = df[(df['测点编号'] == tag) & (df['测量周期'] == interval)& (df[col].abs()<1)][col].sort_values()
                pp_dict['ecdf'] = list(np.arange(1, len(data)+1) / len(data))
                pp_dict['tcdf'] = list(stats.norm.cdf(data, *stats.norm.fit(data)))
                # 执行Shapiro-Wilk检验
                w_stat, p_value = stats.shapiro(data)
    #             print(f"p值 = {p_value:.4f}")
    
                mean = np.mean(data)
                std = np.std(data, ddof=1)
                var = np.var(data, ddof=1)
                Q1, Q2, Q3 = np.percentile(data, [25, 50, 75])
    
                shapiro_dict['测点编号'].append(tag)
                shapiro_dict['测量周期'].append(interval)
                shapiro_dict['单次变形坐标'].append(name)
                shapiro_dict['mean'].append(mean)
                shapiro_dict['std'].append(std)
                shapiro_dict['var'].append(var)
                shapiro_dict['max'].append(max(data))
                shapiro_dict['Q3'].append(Q3)
                shapiro_dict['Q2'].append(Q2)
                shapiro_dict['Q1'].append(Q1)
                shapiro_dict['min'].append(min(data))
                shapiro_dict['p值'].append(p_value)
                shapiro_dict['PP图'].append(pp_dict)
    
                # 结果解读（α=0.05）
                alpha = 0.05
                if p_value > alpha:
                    shapiro_dict['Shapiro-Wilk'].append('服从正态')
    #                 print(f"{i} 不能拒绝原假设：数据服从正态分布 (p > 0.05)")
                else:
                    shapiro_dict['Shapiro-Wilk'].append('不服从正态')
    resultdict['data'] = shapiro_dict
    resultdict['err'] = 0
    return resultdict





def caculate_fangchaqx(df, shapiro_dict,col_dict, intervals, taglist):
    resultdict = {}
    shapiro_df = pd.DataFrame(shapiro_dict['data'])
    fangcha_dict = {'测点编号':[], '测量周期': [],'单次变形坐标':[], 'p值':[], 'Bartlett':[]}
    for name ,col in col_dict.items():  
        for i in taglist: 
            groups = []
            checkdate = []
            for interval in intervals:
                if shapiro_df[(shapiro_df['测点编号'] == i)&(shapiro_df['测量周期'] == interval) & (shapiro_df['Shapiro-Wilk'] == '服从正态')].empty:
                    continue
                groups.append(df[(df['测点编号'] == i) & (df['测量周期'] == interval)& (df[col].abs()<1)][col].to_numpy())
                checkdate.append(interval)
            if len(groups) > 1:
                alpha = 0.05
                stat, p_bartlett = bartlett(*groups)
                if p_bartlett > alpha:
                    fangcha_dict['Bartlett'].append('方差齐性')
        #             print(f"{i} 不能拒绝原假设：数据服从正态分布 (p > 0.05)")
                else:
                    fangcha_dict['Bartlett'].append('方差不齐')
        #             print(f"{i} 拒绝原假设：             数据不服从正态分布 (p ≤ 0.05)")
                fangcha_dict['测点编号'].append(i)
                fangcha_dict['测量周期'].append(checkdate)
                fangcha_dict['单次变形坐标'].append(name)
                fangcha_dict['p值'].append(p_bartlett)
    resultdict['data'] = fangcha_dict
    resultdict['err'] = 0
    return resultdict




def caculate_fangchafx(df, fangcha_dict,col_dict, intervals, taglist):
    resultdict = {}

    fangchadf = pd.DataFrame(fangcha_dict['data'])
    f_oneway_dict = {'测点编号':[], '测量周期': [],'单次变形坐标':[], 'p值':[], '方差分析':[]}
    for name ,col in col_dict.items():
        for point in taglist:
            groups = []
            checkdate = []
            if fangchadf[(fangchadf['测点编号'] == point) & (fangchadf['单次变形坐标'] == name) & (fangchadf['Bartlett'] == '方差齐性')].empty:
                continue
            for n in fangchadf[(fangchadf['测点编号'] == point)& (fangchadf['单次变形坐标'] == name)]['测量周期'].values[0]:
                groups.append(df[(df['测点编号'] == point) & (df['测量周期'] == n)& (df[col].abs()<1)][col].to_numpy())
                checkdate.append(n)
            alpha = 0.05
            stat, p_value = f_oneway(*groups)
            if p_value > alpha:
                f_oneway_dict['方差分析'].append('均值一致')
        #             print(f"{i} 不能拒绝原假设：数据服从正态分布 (p > 0.05)")
            else:
                f_oneway_dict['方差分析'].append('均值不一致')
        #             print(f"{i} 拒绝原假设：             数据不服从正态分布 (p ≤ 0.05)")
            f_oneway_dict['测点编号'].append(point)
            f_oneway_dict['测量周期'].append(checkdate)
            f_oneway_dict['单次变形坐标'].append(name)
            f_oneway_dict['p值'].append(p_value)
    resultdict['data'] = f_oneway_dict
    resultdict['err'] = 0
    return resultdict


    


def caculate_budengfangchafx(df, fangcha_dict,col_dict, intervals, taglist):
    resultdict = {}

    fangchadf = pd.DataFrame(fangcha_dict['data'])

    resultdict = {}
    pg_dict = {'测点编号':[], '测量周期': [],'单次变形坐标':[], 'p值':[], '不等方差分析':[]}
    fangchadf = pd.DataFrame(fangcha_dict['data'])
    
    for name ,col in col_dict.items():
        for point in taglist:
            checkdate = []
            tmpdflist = []
            if fangchadf[(fangchadf['测点编号'] == point) & (fangchadf['单次变形坐标'] == name) & (fangchadf['Bartlett'] == '方差不齐')].empty:
                continue
            for n in fangchadf[(fangchadf['测点编号'] == point)& (fangchadf['单次变形坐标'] == name)]['测量周期'].values[0]:
                tmpdflist.append(df[(df['测点编号'] == point) & (df['测量周期'] == n)& (df[col].abs()<1)][[col,'测量周期']])
                checkdate.append(n)
            tmpdf = pd.concat(tmpdflist, ignore_index = True)
            aov = pg.welch_anova(dv=col, between='测量周期', data=tmpdf)
            alpha = 0.05
            p_value = aov['p-unc'].values[0]
            if p_value > alpha:
                pg_dict['不等方差分析'].append('均值一致')
        #             print(f"{i} 不能拒绝原假设：数据服从正态分布 (p > 0.05)")
            else:
                pg_dict['不等方差分析'].append('均值不一致')
        #             print(f"{i} 拒绝原假设：             数据不服从正态分布 (p ≤ 0.05)")
            pg_dict['测点编号'].append(point)
            pg_dict['测量周期'].append(checkdate)
            pg_dict['单次变形坐标'].append(name)
            pg_dict['p值'].append(p_value)
    resultdict['data'] = pg_dict
    resultdict['err'] = 0
    return resultdict


def caculate_Kruskal(df, shapiro_dict,col_dict, intervals, taglist):
    resultdict = {}
    Kruskal = {'测点编号':[], '测量周期': [],'单次变形坐标':[], 'p值':[], 'Kruskal-Wallis':[]}
    shapiro_df = pd.DataFrame(shapiro_dict['data'])
    for name ,col in col_dict.items():
        for point in taglist:
            # for interval in intervals:
        # 假设周期列是字符串，如“第1期”“第2期”……
            if shapiro_df[(shapiro_df['测点编号'] == point)&(shapiro_df['单次变形坐标'] == name) & (shapiro_df['Shapiro-Wilk'] == '不服从正态')].empty:
                    continue

            groups = {
                period: df.loc[(df['测点编号'] == point) & (df['测量周期'] == period)& (df[col].abs()<1), col]
                        .to_numpy()
                for period in intervals
            }

            alpha = 0.05
            stat, p = kruskal(*groups.values())
            if p > alpha:
                Kruskal['Kruskal-Wallis'].append('中位数一致')
        #             print(f"{i} 不能拒绝原假设：数据服从正态分布 (p > 0.05)")
            else:
                Kruskal['Kruskal-Wallis'].append('中位数不一致')
        #             print(f"{i} 拒绝原假设：             数据不服从正态分布 (p ≤ 0.05)")
            Kruskal['测点编号'].append(point)
            Kruskal['测量周期'].append(intervals.tolist())
            Kruskal['单次变形坐标'].append(name)
            Kruskal['p值'].append(p)
    
    
    resultdict['data'] = Kruskal
    resultdict['err'] = 0
    
    return resultdict

def caculate_predict(df, col_dict, intervals, taglist):
    resultdict = {}
    interval = sorted(intervals)[-1]
    
    axis_df = df[(df['测量周期'] == interval) ]
    bigdate = axis_df['采集时间'].max()
    forecast_list = []
    for name ,axis in col_dict.items():
        for tag in  taglist:
            axis_df_input = axis_df[(axis_df['测点编号'] == tag)][['采集时间', axis]].rename(columns = {'采集时间': 'ds', axis: 'y'})
            axis_df_input['cap'] = axis_df_input['y'].max()
            axis_df_input['floor'] = axis_df_input['y'].min() 
            axis_df_input['work_hours'] = ((axis_df_input['ds'].dt.hour >= 9) & (axis_df_input['ds'].dt.hour <= 18)).astype(int)
            model = Prophet(growth='logistic', changepoint_prior_scale=0.5,daily_seasonality=False)
            model.add_regressor('work_hours')
            model.add_seasonality(name='day', period=2, fourier_order=4) 
            model = model.fit(axis_df_input)
            future = model.make_future_dataframe(periods=24, freq='4H')
            future = future[future['ds'] > bigdate]
            future['cap'] = axis_df_input['y'].max()
            future['floor'] = axis_df_input['y'].min()
            future['work_hours'] = ((future['ds'].dt.hour >= 9) & (future['ds'].dt.hour <= 18)).astype(int)
            forecast = model.predict(future)
            forecast['单次变形坐标'] = name
            forecast['测点编号'] = tag
            forecast_list.append(forecast[['测点编号' ,'单次变形坐标','ds',  'yhat']])
    result = pd.concat(forecast_list, axis=0, ignore_index = True)
    result = result.rename(columns = {'ds': '预测日期', 'yhat': '预测变形量(mm)'})
    # print(result['预测日期'])
    result = result.to_dict('list')
    resultdict['data'] = result
    resultdict['err'] = 0

    return resultdict

    


    