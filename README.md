# EcnuBot_web
<p align="center" width="100%">
<a href="" target="_blank"><img src="https://github.com/RookieDay/EcnuBot_web/blob/main/assets/EcnuBot.png" alt="EcnuBot" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

#### 华东师范大学专属智能客服，接入华东师范大学研发教育领域对话大模型[EduChat](https://github.com/THUDM/ChatGLM3) ，同时支持[ChatGLM3](https://github.com/THUDM/ChatGLM3)、[qwen-max](https://github.com/QwenLM/Qwen)等多种大模型支持，支持对开放问答、作文批改、启发式教学和情感支持等教育特色功能以及各大模型能力。


#### 本代码为Web端实现，另见[EcnuBot](https://github.com/RookieDay/EcnuBot)

## 目录
- [功能介绍](#spiral_notepad-功能介绍)
- [本地部署](#robot-本地部署)
  - [硬件要求](#硬件要求)
  - [下载安装](#下载安装)
  - [部署示例](#部署示例)
- [部分功能展示](#fountain_pen-部分功能展示)
- [未来计划](#construction-未来计划)
- [致谢](#link-致谢)
- [声明](#page_with_curl-声明)

----

## :spiral_notepad: 功能介绍

**⚡ 支持**   
* [x] web端
* [x] 自动问答
* [x] 流式输出
* [x] 切换明暗模式
* [x] 支持 EduChat 大模型
* [x] 支持 qwen-max 大模型
* [x] 支持 千帆 大模型
* [x] 支持 ChatGLM3 大模型
* [x] 更多大模型支持中...

## :robot: 本地部署
### 硬件要求

1. 本项目涉及到的大模型，均已部署到服务器，本地安装依赖即可启动运行

### 下载安装
```bash
git clone git@github.com:RookieDay/EcnuBot_web.git
cd EcnuBot_web
```
4. 安装相关依赖包

```bash
pip install -r requirements.txt
```

### 使用示例
1. 按照config.py文件备注，修改相关配置

2. 运行，启动服务

```bash
python .\main.py
```

## :fountain_pen: 部分功能展示

<details><summary><b>能力菜单</b></summary>

![image](https://github.com/RookieDay/EcnuBot_web/blob/main/assets/Menu.png)

</details>

<details><summary><b>文生图</b></summary>

![image](https://github.com/RookieDay/EcnuBot_web/blob/main/assets/Test.png)

</details>


## :construction: 未来计划

初代EcnuBot主要集成EduChat教育大模型以及其他各大模型支持，随着面向群体以及用户的需求的扩大，从应用性等角度考虑，未来亦着手建设以下功能：

**⚡ 开发**  
* [ ] 学术解析等功能
* [ ] 用户信息分析等功能
* [ ] 文件内容识别处理、总结等功能
* [ ] 自定义角色等功能
* [ ] 插件支持等
* [ ] 数据存储优化等
* [ ] 低成本部署等（模型量化、CPU部署）
* [ ] 更多大模型接入
* [ ] 小程序、web端、手机端支持等多端应用
* [ ] ...... 


## :heart: 致谢

- [EduChat](https://github.com/icalk-nlp/EduChat) 开源支持 
- [gradio](https://github.com/gradio-app/gradio) 开源支持
- [千帆大模型](https://cloud.baidu.com/product/wenxinworkshop) 提供的接口服务 
- [通义千文问模型](https://www.aliyun.com/product/bailian) 提供的接口服务

## :page_with_curl: 声明

本项目仅供研究目的使用，项目开发者对于使用本项目（包括但不限于数据、模型、代码等）所导致的任何危害或损失不承担责任。
