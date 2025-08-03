from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field
import pandas as pd

app = FastAPI(title="Async Points API")

# 全局 DataFrame
df = pd.read_excel('深圳地铁轨道数据.xlsx')
df['测点编号2'] = df['测点编号'].str.split('-').str[0]
df = df[~df['测点编号2'].isin(['kz1', 'kz2', 'kz3', 'kz4'])]

# 提前缓存前 55 行（只读一次，减少每次请求的重复切片）
df_index = df.iloc[:55, :]

#计算每个测点的滑动平均
df = (df.sort_values(by = ['测点编号', '采集时间'], ascending = [True, True])
        .reset_index(drop = True))

window = 5
for col in ['东累计变形量(mm)', '北累计变形量(mm)', '高程累计变形量(mm)']:
    df[f'{col}_mean'] = (
                df
                .groupby('测点编号')[col]
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True))
    
class PointQuery(BaseModel):
    x: float = Field(..., description="东坐标")
    y: float = Field(..., description="北坐标")
    z: float = Field(..., description="高程")
    cate: str = Field(..., description="cate")

@app.get("/tunnel/point")
async def get_points():
    output = (
        df_index[['东坐标', '北坐标', '高程', '测点编号2']]
        .to_dict(orient="split")
    )
    return {"err": 0, "data": output["data"]}


@app.post("/pic/point")
async def get_pic_points(query: PointQuery):
    '''
    输入
    curl -X POST http://localhost:8000/pic/point \     
-H "Content-Type: application/json" \     
-d '{"x":-2.02669720220227, "y":-16.5932312026474, "z": 0.393704432538329,"cate": "jc5"}'
    '''
    mask1 = (
        (df_index.测点编号2 == query.cate)
    )
    
    # if not any(mask1.tolist()):
    #     raise HTTPException(status_code=404, detail="找不到对应测点cate")
    
    df_target = df_index.loc[mask1].copy()
    df_target['diff'] = (
            df_target[['东坐标', '北坐标', '高程']]
            .apply(lambda row: (row['东坐标'] - query.x) +
                              (row['北坐标'] - query.y) +
                              (row['高程'] - query.z), axis=1)
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
    uvicorn.run("myapp:app", host="0.0.0.0", port=8000, reload=True)