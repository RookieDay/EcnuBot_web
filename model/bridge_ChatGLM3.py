import dashscope
import asyncio
import time
from utils import config, user_data

dashscope.api_key = config.config["dashscope_key"]
chatGLM3_data = {
    "messages": [{"role": "system", "content": "You are a helpful assistant."}]
}


def get_resp(user_input, history):
    model_name = "ChatGLM3"
    if not history:
        chatGLM3_data["messages"] = [chatGLM3_data["messages"][0]]
    chatGLM3_data["messages"].append({"role": "user", "content": user_input})
    try:
        response = dashscope.Generation.call(
            model="chatglm3-6b",
            messages=chatGLM3_data["messages"],
        )
        # 返回来的值，前面有换行
        response = response["output"]["text"][2:]

        chatGLM3_data["messages"].append({"role": "assistant", "content": response})
        asyncio.run(user_data.storge_data(user_input, response, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), model_name))
        return response
    except:
        return "ChatGLM3 任务存在问题"
