import os
import time
import pandas as pd

file_name = "user_like.csv"


async def userlike_data(text_prompt, response, liked, local_time):
    if not os.path.exists(file_name):
        with open(file_name, mode="w", encoding="utf-8") as f:
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            user_dict = {
                "question": text_prompt,
                "response": response,
                "liked": liked,
                "date_time": local_time,
            }
            data_header = pd.DataFrame(user_dict, index=[0])
            data_header.to_csv(file_name, index=False)

    else:
        user_QA = pd.read_csv(file_name)
        user_df = pd.DataFrame(
            [[text_prompt, response, liked, local_time]],
            columns=["question", "response", "liked", "date_time"],
            index=[len(user_QA)],
        )
        user_dfNew = pd.concat([user_QA, user_df], ignore_index=True)
        # print(user_dfNew)
        user_dfNew.to_csv(file_name, index=False)


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
