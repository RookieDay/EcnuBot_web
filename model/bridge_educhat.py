import json
import requests

ecnu_data = {"messages": []}
ecnu_scor = {"messages": []}
ecnu_QA = {"messages": []}
ecnu_emo = {"messages": []}

edu_text = {
    # "ECNU 情感 inner": system_prefix_psy_inner,
    "教学": "ECNU 教学",
    "问答": "ECNU 问答",
    "情感": "ECNU 情感",
}


def get_resp(user_input, edu_radio, max_length, top_p, temperature, history):
    if edu_radio == "教学":
        ecnu_data = ecnu_scor
    if edu_radio == "问答":
        ecnu_data = ecnu_QA
    if edu_radio == "情感":
        ecnu_data = ecnu_emo
    if not history:
        ecnu_data["messages"] = []
    ecnu_data["top_p"] = top_p
    ecnu_data["max_tokens"] = max_length
    ecnu_data["temperature"] = temperature
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
        return response
    except:
        response = "EduChat 任务存在问题"
        print(response)
    return response
