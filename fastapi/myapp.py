import json
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
from service import shesd, caculate_yichang, caculate_zhengtai, caculate_fangchaqx, caculate_fangchafx,  caculate_budengfangchafx, caculate_Kruskal, caculate_predict

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
df_predict: pd.DataFrame = pd.DataFrame()
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
    global save_dict
    global df_predict
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
    df = df.rename(columns = {'index' :'测点编号', 'east_coordinate' :'东坐标', 'east_shift':'东单次变形量(mm)',
                              'east_shift_acc':'东累计变形量(mm)', 'north_corrdinate':'北坐标', 'north_shift':'北单次变形量(mm)',
       'north_shift_acc':'北累计变形量(mm)', 'height_coordinate' :'高程',  'height_shift':'高程单次变形量(mm)', 'height_shift_acc':'高程累计变形量(mm)', 'd1' :'水平角(°)', 'd2':'竖直角(°)',
       'd3':'斜距(m)', 'record_time':'采集时间', 'remark':'备注', 'interval' :'测量周期'})

# {'测点编号': [], ''}

    df['测点编号2'] = df['测点编号'].str.split('-').str[0]
    df = df[~df['测点编号2'].isin(['kz1', 'kz2', 'kz3', 'kz4'])]
    df['采集时间'] = pd.to_datetime(df['采集时间'])
    df_predict = df[df['测量周期'] == '第3期'].copy()
    
    # 提前缓存前 55 行（只读一次，减少每次请求的重复切片）
    df_index = df.iloc[:55, :]
    taglist = df_index['测点编号'].tolist()
    intervals = df['测量周期'].unique()
    # 计算异常值，只计算第3期，用于预测
    df_yichang ,df_predict = caculate_yichang(df_predict, col_dict, ['第3期'], taglist, shesd)
    # 计算正态性
    shapiro_dict = caculate_zhengtai(df, col_dict, intervals, taglist)
    # 计算方差齐性
    fangcha_dict = caculate_fangchaqx(df, shapiro_dict,col_dict, intervals, taglist)
    # 方差分析
    f_oneway_dict = caculate_fangchafx(df, fangcha_dict,col_dict, intervals, taglist)
    # 不等方差分析
    pg_dict = caculate_budengfangchafx(df, fangcha_dict,col_dict, intervals, taglist)
    # Kruskal
    kruskal_dict = caculate_Kruskal(df, shapiro_dict ,col_dict, intervals, taglist)


    save_dict['异常值'] = df_yichang
    save_dict['正态'] = shapiro_dict
    save_dict['方差齐性'] = fangcha_dict
    save_dict['方差分析'] = f_oneway_dict
    save_dict['不等方差分析'] = pg_dict
    save_dict['非正态中值分析'] = kruskal_dict
    save_dict['new'] = 1
    # save_dict['预测值'] = predict_dict

    return {
        "msg": "上传成功，已写入全局变量 df",
        "shape": df.shape,
        # "preview": df.head().to_dict("records")
    }
    # print('11111111111')


def get_point_id(x, y, z , cat):
    global df_index

    
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
    global df_index
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
    curl "http://localhost:8000/all_point?x=-2.02669720220227&y=-16.5932312026474&z=0.393704432538329&cat=jc5"
    http://8.136.1.153:8000/all_point?x=-2.02669720220227&y=-16.5932312026474&z=0.393704432538329&cat=jc5
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
    global save_dict

    if '异常值' in save_dict.keys():
        return save_dict['异常值']


    
@app.get("/zhengtai")
async def get_zhengtai():
    global df
    if '正态' in save_dict.keys():
        return save_dict['正态']


@app.get("/fangchaqx")
async def get_fangchaqx():
    global save_dict
    
    if '方差齐性' in save_dict.keys():
        return save_dict['方差齐性']


    
@app.get("/fangchafx")
async def get_fangchafx():
    global save_dict

    if '方差分析' in save_dict.keys():
        return save_dict['方差分析']

        
    

@app.get("/budengfangchafx")
async def get_budengfangchafx():
    global save_dict


    if '不等方差分析' in save_dict.keys():
        return save_dict['不等方差分析']


@app.get("/Kruskal")
async def get_Kruskal():
    global save_dict

    if '非正态中值分析' in save_dict.keys():
        return save_dict['非正态中值分析']
    
@app.get("/predict")
async def get_predict():

    global save_dict
    global df_predict
    global intervals
    global taglist
    
    if ('预测值' in save_dict.keys()) and (save_dict['new'] == 0):
        return save_dict['预测值']
        # 预测
    predict_dict = caculate_predict(df_predict, col_dict, intervals, taglist)
    save_dict['预测值'] = predict_dict
    save_dict['new'] = 0
    return predict_dict

    

# 本地调试
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("myapp:app", host="0.0.0.0", port=8000, reload=True)