import json, requests
from utils import config

qianfan_data = {"messages": []}


def get_access_token():
    token_url = config.config["qianfan_url"]
    API_KEY = config.config["qianfan_ak"]
    SECRET_KEY = config.config["qianfan_sk"]
    params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY,
    }
    access_token = str(
        requests.post(token_url, params=params).json().get("access_token")
    )
    return access_token


def generate_from_baidu_qianfan(url, headers, payload):
    response = requests.request("POST", url, headers=headers, data=payload, stream=True)
    buffer = ""
    for line in response.iter_lines():
        print("line")
        print(line)
        if len(line) == 0:
            continue
        try:
            dec = line.decode().lstrip("data:")
            dec = json.loads(dec)
            incoming = dec["result"]
            buffer += incoming
            yield buffer
        except:
            if ("error_code" in dec) and ("max length" in dec["error_msg"]):
                raise ConnectionAbortedError(dec["error_msg"])  # 上下文太长导致 token 溢出
            elif "error_code" in dec:
                raise RuntimeError(dec["error_msg"])


def get_resp(user_input, top_p, temperature, history):
    if not history:
        qianfan_data["messages"] = []
    qianfan_data["top_p"] = top_p
    qianfan_data["temperature"] = temperature
    try:
        qianfan_data["messages"].append({"role": "user", "content": user_input})
        payload = json.dumps(qianfan_data)
        headers = {"Content-Type": "application/json"}
        url = config.config["qianfan_api"] + get_access_token()
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=payload)
        resp = response.json()["result"]
        qianfan_data["messages"].append({"role": "assistant", "content": resp})
        return resp
    except:
        return "qianfan 任务存在问题"
