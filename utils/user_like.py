import os
import time
import pandas as pd

file_name = "user_like.csv"


async def storge_data(text_prompt, response, liked, local_time):
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
