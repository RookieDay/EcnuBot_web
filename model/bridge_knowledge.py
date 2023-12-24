import os
import base64
import asyncio
import time
import json
import requests
from utils import config, user_data

knowledge_data = {
    "messages": []
}


def get_resp(user_input, knowledge_name, history):
    model_name = "knowledge_base"
    if not history:
        knowledge_data["messages"] = []
    knowledge_data["messages"].append({"role": "user", "content": user_input})
    knowledge_data["kb_name"] = knowledge_name
    try:
        url = "http://127.0.0.1:8002/gradio_chat"
        payload = json.dumps(knowledge_data)
        headers = {"Content-Type": "application/json"}
        # 发送请求
        response = requests.post(url, data=payload, headers=headers)
        response = response.json()["response"]
        knowledge_data["messages"].append({"role": "assistant", "content": response})
        asyncio.run(user_data.storge_data(user_input, response, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), model_name))
        return response
    except:
        return "知识库问答任务存在问题"

def knowledge_get(knowledge_name, upload_file):
    file_dict = {}
    for file_list in upload_file:
        file_path = file_list.name
        file_name = os.path.split(file_path)[1]
        file_type = os.path.splitext(file_path)[1]
        file_dict[file_name] = {}
        with open(file_path, mode="rb") as file:  # b is important -> binary
            fileContent = base64.b64encode(file.read()).decode("utf-8")
        file_dict[file_name]["file_type"] = file_type
        file_dict[file_name]["fileContent"] = fileContent
    print(file_dict)
    params = {
        "kb_name": knowledge_name,
        "file_dict": file_dict,
    }
    url = "http://127.0.0.1:8002/gradio_knowledge"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps(params)
    response = requests.post(url, data=payload, headers=headers)
    resp = response.json()
    return resp