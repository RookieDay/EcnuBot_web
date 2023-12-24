from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn
import json
import datetime
import torch
from fastapi.middleware.cors import CORSMiddleware
from transformers import LlamaForCausalLM, LlamaTokenizer
import requests
import re
from educhat_config import system_prefix_config

# 默认
# # 开放问答
system_prefix = \
"<|system|>"'''你是一个人工智能助手，名字叫EduChat。
- EduChat是一个由华东师范大学开发的对话式语言模型。
EduChat的工具
- Web search: Disable.
- Calculators: Disable.
EduChat的能力
- Inner Thought: Disable.
对话主题
- General: Enable.
- Psychology: Disable.
- Socrates: Disable.'''"</s>"


session = requests.Session()
# 正则提取摘要和链接
title_pattern = re.compile('<a.target=..blank..target..(.*?)</a>')
brief_pattern = re.compile('K=.SERP(.*?)</p>')
link_pattern = re.compile(
    '(?<=(a.target=._blank..target=._blank..href=.))(.*?)(?=(..h=))')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36 Edg/94.0.992.31'}
proxies = {"http": None, "https": None, }


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def find(search_query, page_num=3):
    url = 'https://cn.bing.com/search?q={}'.format(search_query)
    res = session.get(url, headers=headers, proxies=proxies)
    r = res.text
    print('find...in')
    title = title_pattern.findall(r)
    brief = brief_pattern.findall(r)
    link = link_pattern.findall(r)

    # 数据清洗
    clear_brief = []
    for i in brief:
        tmp = re.sub('<[^<]+?>', '', i).replace('\n', '').strip()
        tmp1 = re.sub('^.*&ensp;', '', tmp).replace('\n', '').strip()
        tmp2 = re.sub('^.*>', '', tmp1).replace('\n', '').strip()
        clear_brief.append(tmp2)

    clear_title = []
    for i in title:
        tmp = re.sub('^.*?>', '', i).replace('\n', '').strip()
        tmp2 = re.sub('<[^<]+?>', '', tmp).replace('\n', '').strip()
        clear_title.append(tmp2)
    # return [{'title': "["+clear_title[i]+"]("+link[i][1]+")", 'content':clear_brief[i]}
    #         for i in range(min(page_num, len(brief)))]
    return [{'title': "["+clear_title[i]+"]("+link[i][1]+")", 'content':clear_brief[i]}
                for i in range(min(page_num, len(brief)))]

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/demo")
async def test_domo(request: Request):
    return {}


@app.post("/chat")
async def chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    functionUsed = json_post_list.get('functionUsed')
    max_length = json_post_list.get('max_tokens')
    model_name = json_post_list.get('model')
    temperature = json_post_list.get('temperature')
    top_p = json_post_list.get('top_p')
# 
    if 'prompt' in json_post_list:
        prompt = json_post_list.get('prompt')
# 
    if 'user_QA' in json_post_list:
        user_QA = json_post_list.get('user_QA')
    
    if user_QA in system_prefix_config:
        system_prefix = system_prefix_config[user_QA]
        
    messages = ""
    if "messages" in json_post_list:
        messages = json_post_list.get('messages')
        prompt = messages[-1]["content"]
    
    def talk(history, human_input, max_length, temperature):
        # prefix = ""
        prefix = "<|system|>您好，有什么可以帮助您?</s>"
        
        from enum import Enum

        class ChatRole(str, Enum):
            system = "<|system|>"
            prompter = "<|prompter|>"
            assistant = "<|assistant|>"
        global model, tokenizer
        histories = []
        for question, answer in history:
            histories.append(
                f"{ChatRole.prompter}{question.strip('</s>')}</s>"
                + f"{ChatRole.assistant}{answer.strip('</s>')}</s>"
            )
        if len(histories) > 0:
            prefix += "".join(histories)
            # add sep at the end
        # prefix += f"{ChatRole.prompter}{human_input}</s>{ChatRole.assistant}"
        prefix += f"{ChatRole.prompter}" + ":" + f"{human_input}</s>{ChatRole.assistant}" +":"
        
        prefix = system_prefix + prefix
        
        print(prefix)
        # inputs = tokenizer(prefix, return_tensors="pt", padding=True).to(0)
        inputs = tokenizer(prefix, return_tensors="pt").to(0)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        outputs = model.generate(
            **inputs,
            early_stopping=True,
            max_new_tokens=max_length,
            # do_sample=args.do_sample,
            num_beams=1,
            # top_k=args.top_k,
            top_p=0.7,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            # repetition_penalty=1.01,
        )
        output = tokenizer.decode(outputs[0], truncate_before_pattern=[
                                  r"\n\n^#", "^'''", "\n\n\n"])
        answer = output.split(f"{ChatRole.assistant}")[-1]
        return answer

        
        # return 'answer'

    history = []
    for i in range(max(-11, -len(messages)+1), -1, 2):
        history.append((messages[i]['content'], messages[i+1]['content']))


    print(system_prefix)
    print(user_QA)
        
    if system_prefix == "search":
        print('in....')
        if 'inner' in user_QA:
            prompt = "心理疏导：" + prompt
        # print(prompt)
        search_content = await find(prompt)
        # print(search_content)
        if len(search_content) != 0:
            response = "来自必应搜索解答：\n" + "主题：" +  search_content[-1]["title"] + "\n回答：" + search_content[-1]["content"]
        else:
            response = talk(history, prompt, max_length if max_length else 2048,
                temperature if temperature else 0.95)
    else:
        response = talk(history, prompt, max_length if max_length else 2048,
                        temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    print(answer)
    # log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    # print(log)
    # torch_gc()
    return answer


if __name__ == '__main__':
    
    print('load model.....')
    # model_path = '/root/bigdl/EduChat-main/educhat-sft-002-7b-decrypt'
    model_path = '/root/autodl-tmp/educhat-sft-002-7b-decrypt'
    
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,).cuda()
    print('load model ok.....')
    model.eval()
    uvicorn.run(app, host="127.0.0.1", port=8001, workers=1)

    

#     tokenizer = ''
#     model = ''
#     uvicorn.run(app='educhat_api:app', host="127.0.0.1", port=8001, reload=True)
