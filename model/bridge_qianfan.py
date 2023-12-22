import json, requests
from utils import config, user_data
import asyncio
import time

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
    model_name = "qianfan"
    if not history:
        qianfan_data["messages"] = []
    qianfan_data["top_p"] = top_p
    qianfan_data["temperature"] = temperature

    # response =  "测试先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。诚宜开张圣听，以光先帝遗德，恢弘志士之气，不宜妄自菲薄，引喻失义，以塞忠谏之路也。宫中府中，俱为一体，陟罚臧否，不宜异同。若有作奸犯科及为忠善者，宜付有司论其刑赏，以昭陛下平明之理，不宜偏私，使内外异法也。侍中、侍郎郭攸之、费祎、董允等，此皆良实，志虑忠纯，是以先帝简拔以遗陛下。愚以为宫中之事，事无大小，悉以咨之，然后施行，必能裨补阙漏，有所广益。将军向宠，性行淑均，晓畅军事，试用于昔日，先帝称之曰能，是以众议举宠为督。愚以为营中之事，悉以咨之，必能使行阵和睦，优劣得所。亲贤臣，远小人，此先汉所以兴隆也；亲小人，远贤臣，此后汉所以倾颓也。先帝在时，每与臣论此事，未尝不叹息痛恨于桓、灵也。侍中、尚书、长史、参军，此悉贞良死节之臣，愿陛下亲之信之，则汉室之隆，可计日而待也。臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯。先帝不以臣卑鄙，猥自枉屈，三顾臣于草庐之中，咨臣以当世之事，由是感激，遂许先帝以驱驰。后值倾覆，受任于败军之际，奉命于危难之间，尔来二十有一年矣。先帝知臣谨慎，故临崩寄臣以大事也。受命以来，夙夜忧叹，恐托付不效，以伤先帝之明，故五月渡泸，深入不毛。今南方已定，兵甲已足，当奖率三军，北定中原，庶竭驽钝，攘除奸凶，兴复汉室，还于旧都。此臣所以报先帝而忠陛下之职分也。至于斟酌损益，进尽忠言，则攸之、祎、允之任也。愿陛下托臣以讨贼兴复之效，不效，则治臣之罪，以告先帝之灵。若无兴德之言，则责攸之、祎、允等之慢，以彰其咎；陛下亦宜自谋，以咨诹善道，察纳雅言，深追先帝遗诏，臣不胜受恩感激。今当远离，临表涕零，不知所言。"
    response = "试先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。诚宜开张圣听"
    asyncio.run(user_data.storge_data(user_input, response, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), model_name))
    return response
    try:
        qianfan_data["messages"].append({"role": "user", "content": user_input})
        payload = json.dumps(qianfan_data)
        headers = {"Content-Type": "application/json"}
        url = config.config["qianfan_api"] + get_access_token()
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=payload)
        resp = response.json()["result"]
        qianfan_data["messages"].append({"role": "assistant", "content": resp})
        asyncio.run(user_data.storge_data(user_input, response, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), model_name))
        return resp
    except:
        return "qianfan 任务存在问题"
