import os
import sys

sys.path.append(os.path.abspath(os.curdir))

BASE_PATH = os.getcwd() + "\\themes"


def get_content(file_path, file_name):
    read_path = os.path.join(BASE_PATH, file_path, file_name)
    print(read_path)
    with open(read_path, "r", encoding="utf-8") as f:
        content = f.read()
        print(content)
    print(content)
    return content


dark_mode, likeBtn, blockCss = (
    get_content("js", "dark.js"),
    get_content("js", "likeBtn.js"),
    get_content("css", "block.css"),
)
