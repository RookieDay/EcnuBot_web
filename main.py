import time
import gradio as gr
from model import bridge_qianfan, bridge_ChatGLM3, bridge_educhat, bridge_qwen

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def reset_state():
    return [], [], None


def reset_radio_input():
    return None


def user_text(user_input, history):
    prompt = parse_text(user_input)
    return "", history + [[prompt, None]]


def predict(
    chatbot,
    model_dropdown,
    edu_radio,
    max_length,
    top_p,
    temperature,
):
    history = chatbot[:-1]
    prompt = chatbot[-1][0]
    chatbot[-1][1] = ""
    if model_dropdown == "EduChat":
        response = bridge_educhat.get_resp(
            prompt, edu_radio, max_length, top_p, temperature, history
        )
    if model_dropdown == "qianfan":
        response = bridge_qianfan.get_resp(prompt, top_p, temperature, history)
    if model_dropdown == "qwen":
        response = bridge_qwen.get_resp(prompt, max_length, top_p, temperature, history)
    if model_dropdown == "ChatGLM3":
        response = bridge_ChatGLM3.get_resp(prompt, history)
    history = history + [[prompt, response]]
    for stream_char in response:
        print(stream_char)
        chatbot[-1][1] += stream_char
        time.sleep(0.1)
        yield chatbot


with gr.Blocks() as demo:
    gr.Markdown(
        """\
    <p align="center"><img src='/file=assets/Logo.png' style="height: 60px"/><p>"""
    )
    gr.Markdown("""<center><font size=6>EcnuBot</center>""")
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=1, elem_id="model_settings"):
                model_dropdown = gr.Dropdown(
                    ["EduChat", "qwen", "qianfan", "ChatGLM3"],
                    value="EduChat",
                    label="Model list",
                    interactive=True,
                )
                edu_radio = gr.Radio(
                    ["问答", "情感", "教学"],
                    value="问答",
                    label="Prompt type",
                    visible=True,
                    interactive=True,
                )
                with gr.Accordion("Settings", open=True):
                    max_length = gr.Slider(
                        0,
                        4096,
                        value=1024,
                        step=1.0,
                        label="Maximum length",
                        interactive=True,
                    )
                    top_p = gr.Slider(
                        0, 1, value=0.8, step=0.01, label="Top P", interactive=True
                    )
                    temperature = gr.Slider(
                        0,
                        1,
                        value=0.95,
                        step=0.01,
                        label="Temperature",
                        interactive=True,
                    )
                dark_mode_btn = gr.Button("切换界面明暗模式 ☀", variant="secondary").style(
                    size="sm"
                )

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    avatar_images=["assets/User.png", "assets/EcnuBot.png"]
                )
                user_input = gr.Textbox(
                    show_label=False, placeholder="请输入您的提问", lines=2
                ).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    with gr.Row():
                        emptyBtn = gr.Button("🧹 Clear History (清除历史)")
                        submitBtn = gr.Button("🚀 Submit (发送)", variant="primary")
                        # regen_btn = gr.Button("🤔️ Regenerate (重试)")

    history = gr.State([])

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
        _js="""() => {
                if (document.querySelectorAll('.dark').length) {
                    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                } else {
                    document.querySelector('body').classList.add('dark');
                }
            }""",
    )

    submitBtn.click(
        user_text, [user_input, chatbot], [user_input, chatbot], queue=False
    ).then(
        predict,
        [chatbot, model_dropdown, edu_radio, max_length, top_p, temperature],
        [chatbot],
    )
    edu_radio.select(reset_radio_input, [], [user_input], queue=False)
    emptyBtn.click(reset_state, outputs=[chatbot, history], queue=False)

demo.queue().launch(
    share=False,
    server_name="127.0.0.1",
    server_port=8501,
    inbrowser=True,
    allowed_paths=["./"],
)
