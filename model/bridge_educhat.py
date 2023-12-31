import json
import requests
import asyncio
import time
from utils import user_data

ecnu_data = {"messages": []}
ecnu_scor = {"messages": []}
ecnu_QA = {"messages": []}
ecnu_emo = {"messages": []}

edu_text = {
    # "ECNU 情感 inner": system_prefix_psy_inner,
    "问答-无搜索": "ECNU 问答",
    "问答-有搜索": "ECNU 问答 搜索",
    "教学": "ECNU 教学",
    "情感": "ECNU 情感",
}


def get_resp(user_input, edu_radio, max_length, top_p, temperature, history):
    model_name = "EduChat"
    print('edu_radio')
    print(edu_radio)
    if edu_radio == "问答-无搜索" or edu_radio == "问答-有搜索":
        ecnu_data = ecnu_QA
    if edu_radio == "教学":
        ecnu_data = ecnu_scor
    if edu_radio == "情感":
        ecnu_data = ecnu_emo
    if not history:
        print("in...")
        ecnu_data["messages"] = []
    ecnu_data["top_p"] = top_p
    ecnu_data["max_tokens"] = max_length
    ecnu_data["temperature"] = temperature
    print("ecnu_data")
    print(ecnu_data)
    print(ecnu_QA)
    print(ecnu_scor)
    print(ecnu_emo)
    try:
        url = "http://127.0.0.1:8001/chat"
        ecnu_data["messages"].append({"role": "prompter", "content": user_input})
        ecnu_data["user_QA"] = edu_text[edu_radio]
        payload = json.dumps(ecnu_data)
        headers = {"Content-Type": "application/json"}
        # 发送请求
        response = requests.post(url, data=payload, headers=headers)
        response = response.json()["response"]
        ecnu_data["messages"].append({"role": "assistant", "content": response})
        asyncio.run(
            user_data.storge_data(
                user_input,
                response,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                model_name,
            )
        )
        return response
    except:
        response = "EduChat 任务存在问题"
        print(response)
    return response
