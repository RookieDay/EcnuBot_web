import dashscope
import asyncio
import time
from utils import config, user_data

dashscope.api_key = config.config["dashscope_key"]
tongyi_data = {
    "messages": [{"role": "system", "content": "You are a helpful assistant."}]
}


def get_resp(user_input, max_length, top_p, temperature, history):
    model_name = "qwen-max"
    if not history:
        tongyi_data["messages"] = [tongyi_data["messages"][0]]
    tongyi_data["messages"].append({"role": "user", "content": user_input})
    try:
        response = dashscope.Generation.call(
            model="qwen-max",
            messages=tongyi_data["messages"],
            seed=1234,
            top_p=top_p,
            result_format="message",
            enable_search=False,
            max_tokens=max_length,
            temperature=temperature,
            repetition_penalty=1.0,
            # stream=True,
            # incremental_output=True  # get streaming output incrementally
        )
        response = response["output"]["choices"][0]["message"]["content"]
        tongyi_data["messages"].append({"role": "assistant", "content": response})
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
        return "qwen 任务存在问题"
