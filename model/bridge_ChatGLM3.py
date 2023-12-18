import dashscope
from utils import config

dashscope.api_key = config.config["dashscope_key"]
chatGLM3_data = {
    "messages": [{"role": "system", "content": "You are a helpful assistant."}]
}


def get_resp(user_input, history):
    if not history:
        chatGLM3_data["messages"] = [chatGLM3_data["messages"][0]]
    chatGLM3_data["messages"].append({"role": "user", "content": user_input})
    try:
        response = dashscope.Generation.call(
            model="chatglm3-6b",
            messages=chatGLM3_data["messages"],
        )
        response = response["output"]["text"]

        chatGLM3_data["messages"].append({"role": "assistant", "content": response})
        return response
    except:
        return "ChatGLM3 任务存在问题"
