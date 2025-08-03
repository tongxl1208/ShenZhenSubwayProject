from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

app = FastAPI(title="Async Points API")

class PointsIn(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]
    category: List[str]

    @validator("x", "y", "z", "category")
    def same_lengths(cls, v, values):
        lengths = {len(v)}
        if {"x", "y", "z", "category"} <= values.keys():
            lengths |= {len(values[k]) for k in ["x", "y", "z", "category"]}
        if len(lengths) != 1:
            raise ValueError("x, y, z 与 category 长度必须相等")
        return v

@app.post("/format")
async def echo_points(payload: PointsIn):
    """异步端点，组装返回 [[x, y, z, category], ...]"""
    return [
        [xi, yi, zi, ci]
        for xi, yi, zi, ci in zip(payload.x, payload.y, payload.z, payload.category)
    ]

#  ── 新增的入口判断 ──
if __name__ == "__main__":
    import uvicorn
    # 本地调试：reload=True 方便热更新，正式部署可去掉
    uvicorn.run("myapp:app", host="0.0.0.0", port=8000, reload=True)
