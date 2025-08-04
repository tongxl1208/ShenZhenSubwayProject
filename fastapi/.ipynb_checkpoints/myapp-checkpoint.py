import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field


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
df = pd.read_excel('深圳地铁轨道数据.xlsx')
df['测点编号2'] = df['测点编号'].str.split('-').str[0]
df = df[~df['测点编号2'].isin(['kz1', 'kz2', 'kz3', 'kz4'])]

# 提前缓存前 55 行（只读一次，减少每次请求的重复切片）
df_index = df.iloc[:55, :]

#计算每个测点的滑动平均
df = (df.sort_values(by = ['测点编号', '采集时间'], ascending = [True, True])
        .reset_index(drop = True))

window = 1
for col in ['东累计变形量(mm)', '北累计变形量(mm)', '高程累计变形量(mm)']:
    df[f'{col}_mean'] = (
                df
                .groupby('测点编号')[col]
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True))
    


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


@app.get("/pic/point")
async def get_pic_points(
    x: float = Query(..., description="东坐标"),
    y: float = Query(..., description="北坐标"),
    z: float = Query(..., description="高程"),
    cat: str = Query(..., description="cat")
):
    '''
    输入
    GET 示例：
    curl "http://localhost:8000/pic/point?x=-2.02669720220227&y=-16.5932312026474&z=0.393704432538329&cat=jc5"
    '''
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


    
    # if not any(mask2.tolist()):
    #     raise HTTPException(status_code=404, detail="测点坐标错误")
    

    PointNo = df_target[df_target['diff'] == 0]['测点编号'].values[0]
    df_point = df[df.测点编号 == PointNo][window - 1 :].copy().reset_index(drop=True).reset_index()

    output = (
            df_point[['index' ,'东累计变形量(mm)_mean','北累计变形量(mm)_mean', '高程累计变形量(mm)_mean']]
            .to_dict(orient="list")
        )
    
    return {"err": 0, "x": list(zip(output["index"], output["东累计变形量(mm)_mean"])),
         "y": list(zip(output["index"], output["北累计变形量(mm)_mean"])),
         "z": list(zip(output["index"], output["高程累计变形量(mm)_mean"])),
        }
    

# 本地调试
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("myapp:app", host="0.0.0.0", port=8800, reload=True)