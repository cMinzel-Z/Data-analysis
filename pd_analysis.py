# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:02:17 2020

@author: Minzel
"""

# 需要使用 json 包解析 json 文件
import json
import pandas as pd

def analysis(file, user_id):
    times = 0
    minutes = 0

    # 完成剩余代码
    df = pd.read_json(file,orient='values',encoding='utf-8') 
    times = df[df['user_id'] == user_id].shape[0]
    minutes = df[df['user_id'] == user_id]['minutes'].sum()
    
    return times, minutes