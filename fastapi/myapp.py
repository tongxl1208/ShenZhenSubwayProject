from service import shesd
import json
import time
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


app = FastAPI(title="Async Points API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 或指定前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
# 全局 DataFrame

df: pd.DataFrame = pd.DataFrame()
df_index: pd.DataFrame = pd.DataFrame()
col_dict = {'东' :'东单次变形量(mm)', '北': '北单次变形量(mm)',  '高程':'高程单次变形量(mm)'}
yichang_dict = {}

intervals: list = []
taglist: list = []

save_dict : dict = {}

class Payload(BaseModel):
    payload: List[Any]

@app.post("/upload")
async def upload_points(payload: Payload):
    """
    请求体示例：
    {
        "测点编号": ["A1", "A2", "A3"],
        "东坐标":   [120.1, 120.2, 120.3]
    }
    """
    global df
    global intervals
    global df_index
    global taglist
    try:
        # 直接利用字典构造 DataFrame
        data = payload.dict()
        df = pd.DataFrame(data['payload'])
        
        if df.empty:
            raise ValueError("空数据")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析失败: {e}")

    # 覆盖全局变量
    # GLOBAL_DF = df.copy()
    df.columns = ['测点编号', '东坐标', '东单次变形量(mm)', '东累计变形量(mm)', '北坐标', '北单次变形量(mm)',
       '北累计变形量(mm)', '高程', '高程单次变形量(mm)', '高程累计变形量(mm)', '水平角(°)', '竖直角(°)',
       '斜距(m)', '采集时间', '备注', '测量周期']

# {'测点编号': [], ''}

    df['测点编号2'] = df['测点编号'].str.split('-').str[0]
    df = df[~df['测点编号2'].isin(['kz1', 'kz2', 'kz3', 'kz4'])]
    df['采集时间'] = pd.to_datetime(df['采集时间'])
    
    # 提前缓存前 55 行（只读一次，减少每次请求的重复切片）
    df_index = df.iloc[:55, :]
    taglist = df_index['测点编号'].tolist()
    intervals = df['测量周期'].unique()

    return {
        "msg": "上传成功，已写入全局变量 df",
        "shape": df.shape,
        # "preview": df.head().to_dict("records")
    }
    # print('11111111111')


def get_point_id(x, y, z , cat):
    mask1 = (
        (df_index.测点编号2 == cat)
    )
    
    # if not any(mask1.tolist()):
    #     raise HTTPException(status_code=404, detail="找不到对应测点cat")
    
    df_target = df_index.loc[mask1].copy()
    df_target['diff'] = (
            df_target[['东坐标', '北坐标', '高程']]
            .apply(lambda row: (row['东坐标'] - x) +
                              (row['北坐标'] - y) +
                              (row['高程'] - z), axis=1)
    )

    PointNo = df_target[df_target['diff'] == 0]['测点编号'].values[0]
    return PointNo

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/tunnel/point")
async def get_points():
    output = (
        df_index[['东坐标', '北坐标', '高程', '测点编号2']]
        .to_dict(orient="split")
    )
    return {"err": 0, "data": output["data"]}



@app.get("/all_point")
async def get_pic_points(
    x: float = Query(..., description="东坐标"),
    y: float = Query(..., description="北坐标"),
    z: float = Query(..., description="高程"),
    cat: str = Query(..., description="cat")
):
    '''
    输入
    GET 示例：
    curl "http://localhost:8800/all_point?x=-2.02669720220227&y=-16.5932312026474&z=0.393704432538329&cat=jc5"
    http://8.136.1.153:8800/all_point?x=-2.02669720220227&y=-16.5932312026474&z=0.393704432538329&cat=jc5
    '''
    PointNo = get_point_id(x, y, z , cat)
    df_point = df[df.测点编号 == PointNo].copy().reset_index(drop=True).reset_index()

    output = (
            df_point[['测点编号' ,'东累计变形量(mm)','北累计变形量(mm)', '高程累计变形量(mm)']]
            .to_dict(orient="list")
        )
    
    return {"err": 0, "x": list(zip(output["测点编号"], output["东累计变形量(mm)"])),
         "y": list(zip(output["测点编号"], output["北累计变形量(mm)"])),
         "z": list(zip(output["测点编号"], output["高程累计变形量(mm)"])),
        }
    


@app.get("/yichang")
async def get_yichang():
    global df
    global save_dict

    if '异常值' in save_dict.keys():
        return save_dict['异常值']

    df_yichang = {'测量周期': [], '单次变形坐标': [], '检测点': [], 'P值':[], '异常值数量': []}
    resultdict = {}
    axis_dfs = []
    alpha = 0.025
    for name ,col in col_dict.items():
        count = 0
        for interval in intervals:
            for tag in taglist:
                R = df[(df['测点编号'] == tag) & (df['测量周期'] == interval)][col].to_numpy()
                mean_val = np.mean(R)
                res = shesd(R, period=None, max_anoms=None, alpha=alpha)
                R[res["anomaly_indices"]] = mean_val
                mask = (df['测点编号'] == tag) & (df['测量周期'] == interval)
                df.loc[mask, col] = R
                count += len(res["anomaly_indices"])                
                df_yichang['测量周期'].append(interval)
                df_yichang['单次变形坐标'].append(name)
                df_yichang['检测点'].append(tag)
                df_yichang['P值'].append(alpha)
                df_yichang['异常值数量'].append(len(res["anomaly_indices"]))
    resultdict['data'] = df_yichang
    resultdict['err'] = 0
    save_dict['异常值'] = resultdict
    # df.to_excel('check.xlsx', index = False)

    
    return resultdict


    
@app.get("/zhengtai")
async def get_zhengtai():
    global save_dict
    global df
    print(df.to_excel('df.xlsx', index = False))
    if '正态' in save_dict.keys():
        return save_dict['正态']

    resultdict = {}

    shapiro_dict = {'测点编号':[], '测量周期': [],'单次变形坐标':[],'mean':[], 'std': [],'var': [], 
                'max': [], 'Q3': [], 'Q2':[], 'Q1':[], 'min': [], 'p值':[], 'Shapiro-Wilk':[], 'PP图':[]}
    
    for name ,col in col_dict.items():
        for interval in intervals:
            for tag in taglist:
                pp_dict = {}
                data = df[(df['测点编号'] == tag) & (df['测量周期'] == interval)][col].sort_values()
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
    save_dict['正态'] = resultdict
    return resultdict



@app.get("/fangchaqx")
async def get_fangchaqx():
    global save_dict
    global df
    if '方差齐性' in save_dict.keys():
        return save_dict['方差齐性']

    resultdict = {}

    shapiro_df = pd.DataFrame(save_dict['正态']['data'])
    print(shapiro_df)

    fangcha_dict = {'测点编号':[], '测量周期': [],'单次变形坐标':[], 'p值':[], 'Bartlett':[]}
    for name ,col in col_dict.items():  
        for i in taglist: 
            groups = []
            checkdate = []
            for interval in intervals:
                if shapiro_df[(shapiro_df['测点编号'] == i)&(shapiro_df['测量周期'] == interval) & (shapiro_df['Shapiro-Wilk'] == '服从正态')].empty:
                    continue
                groups.append(df[(df['测点编号'] == i) & (df['测量周期'] == interval)][col].to_numpy())
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

    save_dict['方差齐性'] = resultdict
    return resultdict




    
@app.get("/fangchafx")
async def get_fangchafx():
    global save_dict
    global df

    if '方差分析' in save_dict.keys():
        return save_dict['方差分析']

    resultdict = {}

    fangchadf = pd.DataFrame(save_dict['方差齐性']['data'])
    f_oneway_dict = {'测点编号':[], '测量周期': [],'单次变形坐标':[], 'p值':[], '方差分析':[]}
    for name ,col in col_dict.items():
        for point in taglist:
            groups = []
            checkdate = []
            if fangchadf[(fangchadf['测点编号'] == point) & (fangchadf['单次变形坐标'] == name) & (fangchadf['Bartlett'] == '方差齐性')].empty:
                continue
            for n in fangchadf[(fangchadf['测点编号'] == point)& (fangchadf['单次变形坐标'] == name)]['测量周期'].values[0]:
                groups.append(df[(df['测点编号'] == point) & (df['测量周期'] == n)][col].to_numpy())
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
    save_dict['方差分析'] = resultdict
    return resultdict
        
    

@app.get("/budengfangchafx")
async def get_budengfangchafx():
    global save_dict
    global df

    if '不等方差分析' in save_dict.keys():
        return save_dict['不等方差分析']

    resultdict = {}
    pg_dict = {'测点编号':[], '测量周期': [],'单次变形坐标':[], 'p值':[], '不等方差分析':[]}
    fangchadf = pd.DataFrame(save_dict['方差齐性']['data'])
    
    for name ,col in col_dict.items():
        for point in taglist:
            checkdate = []
            tmpdflist = []
            if fangchadf[(fangchadf['测点编号'] == point) & (fangchadf['单次变形坐标'] == name) & (fangchadf['Bartlett'] == '方差不齐')].empty:
                continue
            for n in fangchadf[(fangchadf['测点编号'] == point)& (fangchadf['单次变形坐标'] == name)]['测量周期'].values[0]:
                tmpdflist.append(df[(df['测点编号'] == point) & (df['测量周期'] == n)][[col,'测量周期']])
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

        
    save_dict['不等方差分析'] = resultdict
    return resultdict


@app.get("/Kruskal")
async def get_Kruskal():
    global save_dict
    global df

    if '非正态中值分析' in save_dict.keys():
        return save_dict['非正态中值分析']
    resultdict = {}
    Kruskal = {'测点编号':[], '测量周期': [],'单次变形坐标':[], 'p值':[], 'Kruskal-Wallis':[]}
    shapiro_df = pd.DataFrame(save_dict['正态']['data'])
    for name ,col in col_dict.items():
        for point in taglist:
            for interval in intervals:
        # 假设周期列是字符串，如“第1期”“第2期”……
                if shapiro_df[(shapiro_df['测点编号'] == point)&(shapiro_df['测量周期'] == interval) & (shapiro_df['Shapiro-Wilk'] == '不服从正态')].empty:
                        continue
    
                groups = {
                    period: df.loc[(df['测点编号'] == point) & (df['测量周期'] == period), col]
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
                break
    
    resultdict['data'] = Kruskal
    resultdict['err'] = 0
    save_dict['非正态中值分析'] = resultdict
    
    return resultdict
    
@app.get("/predict")
async def get_predict():

    global save_dict
    global df

    if '预测值' in save_dict.keys():
        return save_dict['预测值']
    resultdict = {}
    
    axis_df = df[(df['测量周期'] == '第3期') ]
    forecast_list = []
    for name ,axis in col_dict.items():
        for tag in  taglist:
            axis_df_input = axis_df[(axis_df['测点编号'] == tag)][['采集时间', axis]].rename(columns = {'采集时间': 'ds', axis: 'y'})
            axis_df_input['cap'] = axis_df_input['y'].max()
            axis_df_input['floor'] = axis_df_input['y'].min() 
            axis_df_input['work_hours'] = ((axis_df_input['ds'].dt.hour >= 9) & (axis_df_input['ds'].dt.hour <= 18)).astype(int)
            train, test = axis_df_input[:42], axis_df_input[42:]
            # 预测模型
            model = Prophet(growth='logistic', changepoint_prior_scale=0.5,daily_seasonality=False)
            model.add_regressor('work_hours')
            model.add_seasonality(name='day', period=2, fourier_order=4) 
            model = model.fit(train)
            forecast = model.predict(test)
            forecast['y'] = test['y'].tolist()
            forecast['单次变形坐标'] = name
            forecast['测点编号'] = tag
            forecast_list.append(forecast[['测点编号' ,'单次变形坐标','ds',  'yhat']])
    result = pd.concat(forecast_list, axis=0, ignore_index = True)
    result = result.rename(columns = {'ds': '预测日期', 'yhat': '预测变形量(mm)'})
    result = result.to_dict('list')
    resultdict['data'] = result
    resultdict['err'] = 0
    save_dict['预测值'] = resultdict
    return resultdict

    

# 本地调试
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("myapp:app", host="0.0.0.0", port=8000, reload=True)