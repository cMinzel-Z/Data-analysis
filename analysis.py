# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:45:07 2019

@author: Minzel
"""

# 需要使用 json 包解析 json 文件
import json


def analysis(file, user_id):
    times = 0
    minutes = 0

    try:
        with open(file) as f:
            data = json.load(f)

        for i in data:
            if i['user_id'] == user_id:
                times += 1
                minutes += i['minutes']
    
    except:
        return 0
        
    return times, minutes


if __name__ == '__main__':
    a = analysis('user_study.json', 199071)
    print(a)