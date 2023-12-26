import time
import asyncio
import gradio as gr
from model import (
    bridge_qianfan,
    bridge_ChatGLM3,
    bridge_educhat,
    bridge_qwen,
    bridge_knowledge,
)
from themes.base import dark_mode, likeBtn, blockCss
from utils import gradio_utils


def reset_state():
    return [], [], None


def reset_radio_input():
    return None


def user_text(user_input, history):
    if user_input == "":
        return user_input, user_input, history
    prompt = gradio_utils.parse_text(user_input)

    return gr.update(interactive=False), gr.update(value=""), history + [[prompt, None]]


def text_unlock(k):
    return gr.update(interactive=True)


def regenerate(user_input, chatbot):
    if not chatbot:
        yield gr.update(interactive=False), chatbot
        return
    chatbot[-1][1] = None
    yield gr.update(interactive=False), chatbot
    return


def vote(chatbot, index_state, data: gr.LikeData):
    value_new = data.value
    index_new = data.index
    asyncio.run(
        gradio_utils.userlike_data(
            chatbot[index_new[0]][0],
            chatbot[index_new[0]][1],
            data.liked,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
    )
    if len(index_state) == 0:
        index_state.append(index_new)
    else:
        if index_new in index_state:
            index_state.pop(-1)
            index_state.append(index_new)
    return index_state


def predict(
    chatbot,
    quest_select,
    knowledge_name,
    knowledge_dropdown,
    model_dropdown,
    edu_radio,
    max_length,
    top_p,
    temperature,
):
    if not chatbot or ((len(chatbot) >= 1 and chatbot[-1][1])):
        yield chatbot
        return
    history = chatbot[:-1]
    prompt = chatbot[-1][0]
    chatbot[-1][1] = ""
    if quest_select == "知识库问答":
        if len(knowledge_dropdown) >= 1:
            print("n。。")
            print(knowledge_dropdown)
            response = bridge_knowledge.get_resp(prompt, knowledge_dropdown, history)
        response = bridge_knowledge.get_resp(prompt, knowledge_name, history)
    if quest_select == "模型问答":
        if model_dropdown == "EduChat":
            response = bridge_educhat.get_resp(
                prompt, edu_radio, max_length, top_p, temperature, history
            )
        if model_dropdown == "qianfan":
            response = bridge_qianfan.get_resp(prompt, top_p, temperature, history)
        if model_dropdown == "qwen":
            response = bridge_qwen.get_resp(
                prompt, max_length, top_p, temperature, history
            )
        if model_dropdown == "ChatGLM3":
            response = bridge_ChatGLM3.get_resp(prompt, history)
    history = history + [[prompt, response]]
    print("history")
    print(history)
    for stream_char in response:
        # print(stream_char)
        chatbot[-1][1] += stream_char
        time.sleep(0.05)
        yield chatbot


with gr.Blocks(title="EcnuBot", css="" + blockCss + "") as demo:
    gr.Markdown(
        """\
    <p align="center"><img src='/file=assets/Logo.png' style="height: 60px"/><p>"""
    )
    gr.Markdown("""<center><font size=6>EcnuBot</center>""")
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=1, elem_id="model_settings"):
                with gr.Row():
                    quest_select = gr.Radio(
                        ["模型问答", "知识库问答"],
                        value="模型问答",
                        label="对话形式",
                        visible=True,
                        interactive=True,
                    )
                with gr.Row():
                    upload_file = gr.File(
                        label="上传文件",
                        file_types=[".pdf", ".xlsx", "text", ".docx", ".pptx"],
                        file_count="multiple",
                        type="file",
                        visible=False,
                    )
                    knowledge_name = gr.Textbox(
                        label="知识库名称",
                        visible=False,
                        info="重新构建知识库，将清除原知识库问答历史记录",
                    )
                    knowledge_button = gr.Button(
                        value="构建知识库",
                        variant="primary",
                        visible=False,
                    )
                    knowledge_dropdown = gr.Dropdown(
                        label="请选择当前已构建知识库，直接问答提问",
                        visible=False,
                    )

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        ["EduChat", "qwen", "qianfan", "ChatGLM3"],
                        value="EduChat",
                        label="模型列表",
                        visible=True,
                        interactive=True,
                    )
                    edu_radio = gr.Radio(
                        ["问答-无搜索", "问答-有搜索", "情感", "教学"],
                        value="问答-无搜索",
                        label="交互类型",
                        visible=True,
                        interactive=True,
                    )
                with gr.Accordion("设置", open=True) as settings_model:
                    max_length = gr.Slider(
                        0,
                        4096,
                        value=1024,
                        step=1.0,
                        label="Maximum length",
                        visible=True,
                        interactive=True,
                    )
                    top_p = gr.Slider(
                        0,
                        1,
                        value=0.8,
                        step=0.01,
                        label="Top P",
                        interactive=True,
                        visible=True,
                    )
                    temperature = gr.Slider(
                        0,
                        1,
                        value=0.95,
                        step=0.01,
                        label="Temperature",
                        interactive=True,
                        visible=True,
                    )
                dark_mode_btn = gr.Button("切换界面明暗模式 ☀", variant="secondary").style(
                    size="sm"
                )

            with gr.Column(scale=6):
                with gr.Row():
                    chatbot = gr.Chatbot(
                        avatar_images=["assets/User.png", "assets/EcnuBot.png"],
                        show_copy_button=True,
                    )
                with gr.Row():
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="请输入问题，可通过Shift+Enter发送问题",
                        lines=2,
                        interactive=True,
                    ).style(container=False)
                with gr.Row():
                    emptyBtn = gr.Button("🧹 Clear History (清除历史)")
                    submitBtn = gr.Button("🚀 Submit (发送)", variant="primary")
                    retryBtn = gr.Button("🤔️ Regenerate (重试)")
    history = gr.State([])
    index_state = gr.State([])
    knowledge_state = gr.State([])

    def on_select_changed(
        quest_select,
        model_dropdown,
        edu_radio,
        upload_file,
        knowledge_name,
        knowledge_dropdown,
        knowledge_button,
    ):
        model_dropdown = gr.Dropdown.update(visible=True)
        edu_radio = gr.Radio.update(visible=True)
        upload_file = gr.File.update(visible=False)
        knowledge_name = gr.Textbox.update(visible=False)
        knowledge_dropdown = gr.Textbox.update(visible=False)
        knowledge_button = gr.Button.update(visible=False)
        settings_model = gr.Accordion.update(visible=True)

        if not quest_select == "模型问答":
            print("in........")
            print(quest_select)
            model_dropdown = gr.Dropdown.update(visible=False)
            edu_radio = gr.Radio.update(visible=False)
            #
            upload_file = gr.File.update(visible=True)
            knowledge_name = gr.Textbox.update(visible=True)
            print("knowledge_dropdown")
            print(knowledge_dropdown)
            if knowledge_dropdown == None:
                knowledge_dropdown = gr.Button.update(visible=True)
            knowledge_button = gr.Textbox.update(visible=True)
            settings_model = gr.Accordion.update(visible=False)
        return (
            model_dropdown,
            edu_radio,
            upload_file,
            knowledge_name,
            knowledge_dropdown,
            knowledge_button,
            settings_model,
        )

    quest_select.change(
        on_select_changed,
        [
            quest_select,
            model_dropdown,
            edu_radio,
            upload_file,
            knowledge_name,
            knowledge_dropdown,
            knowledge_button,
        ],
        [
            model_dropdown,
            edu_radio,
            upload_file,
            knowledge_name,
            knowledge_dropdown,
            knowledge_button,
            settings_model,
        ],
    )

    def on_md_dropdown_changed(k):
        ret = {edu_radio: gr.update(visible=True)}
        if not k == "EduChat":
            ret = {edu_radio: gr.update(visible=False)}
        return ret

    model_dropdown.change(on_md_dropdown_changed, [model_dropdown], [edu_radio])

    dark_mode_btn.click(
        None,
        None,
        None,
        _js="() => {" + dark_mode + "}",
    )

    submitBtn.click(
        user_text,
        [user_input, chatbot],
        [user_input, user_input, chatbot],
        queue=False,
    ).then(
        predict,
        [
            chatbot,
            quest_select,
            knowledge_name,
            knowledge_dropdown,
            model_dropdown,
            edu_radio,
            max_length,
            top_p,
            temperature,
        ],
        [chatbot],
        show_progress=True,
    ).then(
        text_unlock, [], [user_input]
    ).then(
        None,
        [chatbot],
        None,
        _js="() => {" + likeBtn + "}",
    )

    user_input.submit(
        user_text, [user_input, chatbot], [user_input, user_input, chatbot], queue=False
    ).then(
        predict,
        [
            chatbot,
            quest_select,
            knowledge_name,
            knowledge_dropdown,
            model_dropdown,
            edu_radio,
            max_length,
            top_p,
            temperature,
        ],
        [chatbot],
        show_progress=True,
    ).then(
        text_unlock, [], [user_input]
    ).then(
        None,
        [chatbot],
        None,
        _js="() => {" + likeBtn + "}",
    )

    retryBtn.click(
        regenerate,
        [user_input, chatbot],
        [user_input, chatbot],
        show_progress=True,
    ).then(
        predict,
        [
            chatbot,
            quest_select,
            knowledge_name,
            knowledge_dropdown,
            model_dropdown,
            edu_radio,
            max_length,
            top_p,
            temperature,
        ],
        [chatbot],
        show_progress=True,
    ).then(
        text_unlock, [], [user_input]
    ).then(
        None,
        [chatbot],
        None,
        _js="() => {" + likeBtn + "}",
    )

    def knowledge_create(knowledge_name, upload_file, knowledge_state):
        if upload_file == None:
            gr.Warning("请先上传文件，再点击构建知识库")
            return knowledge_name
        gr.Info("知识库创建中...")
        resp = bridge_knowledge.knowledge_get(knowledge_name, upload_file)
        if resp["status"] == 200:
            print("Connection established. Receiving data...")
            gr.Info("知识库创建成功")
            knowledge_state.insert(0, knowledge_name)
            return knowledge_name
        else:
            print("Failed to connect. Status code:", resp["status"])
            gr.Error("知识库创建失败")
            return knowledge_name

    def show_knowledge_dropdown(knowledge_state):
        if len(knowledge_state) == 0:
            return ""
        return gr.Dropdown.update(
            visible=True, choices=knowledge_state, value=knowledge_state[0]
        )

    knowledge_button.click(
        knowledge_create,
        [knowledge_name, upload_file, knowledge_state],
        [knowledge_name],
    ).then(show_knowledge_dropdown, [knowledge_state], [knowledge_dropdown])

    knowledge_dropdown.change(reset_state, outputs=[chatbot, history], queue=False)

    chatbot.like(vote, [chatbot, index_state], [index_state])

    edu_radio.select(reset_radio_input, [], [user_input], queue=False)
    emptyBtn.click(reset_state, outputs=[chatbot, history], queue=False)

demo.queue().launch(
    share=False,
    favicon_path="./assets/Logo.png",
    server_name="127.0.0.1",
    server_port=8501,
    inbrowser=True,
    allowed_paths=["./"],
    # auth=[("a", "1"), ("b", "2")],
)
