import time
import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    api_key="sk-DcTyNztpGYGOnslp1VCvcnp4ZuJ1giQvwn96IKBqbe1HP9Fg",
    base_url="https://api.moonshot.cn/v1"
)
model = "moonshot-v1-128k"

history = []


def upload(path):
    file_object = client.files.create(file=Path(path), purpose="file-extract")
    file_content = client.files.content(file_id=file_object.id).text
    return file_object, file_content


def chat(query, history):
    history.append({
        "role": "user",
        "content": query
    })
    completion = client.chat.completions.create(
        model=model,
        messages=history,
        temperature=0.3,
    )
    result = completion.choices[0].message.content
    history.append({
        "role": "assistant",
        "content": result
    })
    return result


def chatWithFile(filePath, query):
    _, file_content = upload(filePath)
    message_list = []
    message_list.append({
        "role": "system",
        "content": file_content
    })
    message_list.append({
        "role": "user",
        "content": query
    })
    time.sleep(3)
    completion = client.chat.completions.create(
        model=model,
        messages=message_list,
        temperature=0.3,
        max_tokens=4096
    )
    result = completion.choices[0].message.content
    return result


def main():
    result1 = chatWithFile("data/2404.00405v1.pdf",
                           "我需要依照这篇论文的方法思路去回顾别的论文，以此梳理总结出一个大语言模型与人类交互的文献综述，论文中提到需要将文献中交互发生的时候分为四个阶段，分别是"
                           "（1）Planning (before an interaction)；"
                           "（2）Facilitating (during an interaction)；"
                           "（3）Iterating (refining an established interaction)；"
                           "（4）Testing (testing a defined interaction)，"
                           "以及论文数据中分类类别一共有11一种，分别为："
                           "Text-based Conversational Prompting；"
                           "Text-based Conversational Prompting with Reasoning；"
                           "UI for Structured Prompts Input；"
                           "UI for Varying Output；"
                           "UI for Iteration of Interaction；"
                           "UI for Testing of Interaction；"
                           "UI for Reasoning；"
                           "Explicit Context；"
                           "Implicit Context；"
                           "Team Process Facilitating；"
                           "Capability-aware Task Delegation."
                           "这篇论文里面有对每一个阶段和每一个类别的详细介绍，如果你能理解我的问题，请你重新复述一遍我的问题.")
    history.append(result1)
    print("A1:", result1)
    print("============================================================================================")
    time.sleep(3)
    result2 = chatWithFile("data/2305.11790v3.pdf",
                           "请你对这篇论文进行分类."
                           "输出格式为："
                           "（1）论文是否与我们的研究主题相关（参考《A Taxonomy for Human-LLM Interaction Modes: An Initial Exploration》）以及理由；"
                           "（2）11个类别中主类别；"
                           "（3）如果有请给出11个类别中的副类别；"
                           "（4）论文中交互发生的主要阶段；"
                           "（5）如果有请给出论文中交互发生的次要阶段；"
                           "（6）这么分类的原因（必须严格贴合原文）；如果没有副分类请直接给出没有，没有凭空捏造或者过度归纳。")
    print("A2:", result2)
    history.append(result2)



if __name__ == '__main__':
    main()
