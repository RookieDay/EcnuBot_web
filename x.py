import gradio as gr
import random

def vote(tmp, index_state, data: gr.LikeData):
    value_new = data.value
    index_new = data.index
    if len(index_state) == 0 :
        index_state.append(index_new)
    else:
        if index_new in index_state:
            return "Your feedback is already saved", index_state
        else:
            index_state.append(index_new)
    return str(data.value) + ";" + str(data.index)+";"+ str(data.liked)+";"+str(index_state), index_state

with gr.Blocks() as demo:
    tmp = gr.Textbox(visible=True, value="")
    chatbot = gr.Chatbot(layout='panel')
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    index_state = gr.State(value=[])
    
    def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    chatbot.like(vote, [tmp, index_state], [tmp, index_state])

demo.launch()