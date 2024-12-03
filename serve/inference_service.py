# 运行
# pip install gradio
# python demo.py
import os 
import sys 
sys.path.append('.')
from typing import List
import gradio as gr
import time

import argparse 
import yaml
from inference.inference.inference_engine import InferenceEngine
from inference.inference.recflow_script.rank_stage import RankerInferenceEngine
from inference.inference.recflow_script.retrieve_stage import RetrieverInferenceEngine
import pandas as pd 
import numpy as np 
from tqdm import tqdm

# 模拟多阶段检索过程
def retrieve(user_id, user_clicks):
    """
    Args:
        user_id (int): 用户ID
        user_clicks (str): 用户点击的商品ID字符串, 以空格分隔
        inference_engine_list (list[InferenceEngine]): 推理引擎列表
    Returns:
        results (List[str]): 每个阶段的检索结果
        final_result (List[Dict]): 最终展示的结果
    """
    global inference_engine_list, user_id2latest_timestamp
    # latest_timestamp = _get_user_latest_timestamp(inference_engine_list[0].redis_client, user_id)
    latest_timestamp = user_id2latest_timestamp[user_id]

    batch_infer_df = pd.DataFrame({
        "user_id": [user_id],
        "request_timestamp": [latest_timestamp],
    })
    
    batch_outputs_list = []
    batch_outputs_df = None 
    for inference_engine in inference_engine_list:
        if inference_engine.config['stage'] == 'retrieve':           
            batch_outputs = inference_engine.batch_inference(batch_infer_df=batch_infer_df)
        else:
            batch_outputs = inference_engine.batch_inference(batch_infer_df=batch_infer_df, 
                                                                batch_candidates_df=batch_outputs_df)
        batch_outputs_list.append(batch_outputs[0])    
        batch_outputs_df = pd.DataFrame({inference_engine.feature_config['fiid']: batch_outputs.tolist()})


    results = []
    final_result = []
    for batch_outputs, inference_engine in zip(batch_outputs_list, inference_engine_list):
        results_str = f'Stage {inference_engine.config["stage"]} results: {batch_outputs.tolist()}'
        results.append(results_str)
    for item in batch_outputs_list[-1]:
        final_result.append({
            "id": int(item),
            "name": f"Video-{item}",
        })

    return results, final_result

def chat_interface(user_clicks, chat_history, user_id):
    # request 按钮触发的动作
    # 获取多阶段检索的中间结果和最终结果（返回一组ID）
    stages_results, final_results = retrieve(user_id, user_clicks)
    
    # 为每个阶段添加折叠展示
    stage_messages = [f"Recommendation Result of User {user_id}:"]
    for i, result in enumerate(stages_results):
        stage_messages.append(f"<details><summary>Stage {i+1}</summary><p>{result}</p></details>")
    
    # 更新对话历史
    user_content = f"Please recommend some items to user {user_id}."
    chat_history.append(
        {"role": "user", 
         "content": "\n".join(["User click: " + user_clicks, user_content]) if user_clicks != '' else user_content})
    chat_history.append({"role": "assistant", "content": "\n".join(stage_messages)})
    
    # 更新展示内容
    textbox_values = [f"item-{item['id']}: {item['name']}" for item in final_results]
    textbox_update_list = []
    for value in textbox_values:
        textbox_update_list.append(gr.update(value=value, visible=True))

    click_bttn_list = [gr.update(visible=True) for i in range(len(final_results))]

    user_clicks = "" # 每次请求，清空上次的点击历史。因为默认历史已入库保留

    # 返回聊天历史和按钮内容
    return chat_history, user_clicks, final_results, *textbox_update_list, *click_bttn_list
    
# 处理用户点击的按钮
def button_click_fns(i):
    def fn(user_history, show_items):
        # 模拟用户点击某个按钮的ID，收集这个ID作为新的用户输入
        user_history = " ".join([user_history, str(show_items[int(i)]["id"])]).strip()
        return user_history
    return fn

def change_user_fns(user_id):
    # 清空对话框并添加一条新消息
    stages_results, final_results = retrieve(user_id, "")
    stage_messages = [f"Recommendation Result of User {user_id}:"]
    for i, result in enumerate(stages_results):
        stage_messages.append(f"<details><summary>Stage {i+1}</summary><p>{result}</p></details>")
    
    # 更新对话历史
    chat_history = []
    chat_history.append({"role": "assistant", "content": f"🤗 Hello, this is user {user_id}"})
    chat_history.append({"role": "assistant", "content": "\n".join(stage_messages)})


    user_clicks = "" # 每次请求，清空上次的点击历史。因为默认历史已入库保留
    send_btn_update = gr.update(interactive=True) # set send button to interactive

    # 更新展示内容
    textbox_values = [f"item-{item['id']}: {item['name']}" for item in final_results]
    textbox_update_list = []
    for value in textbox_values:
        textbox_update_list.append(gr.update(value=value, visible=True))

    click_bttn_list = [gr.update(visible=True) for i in range(len(final_results))]

    return chat_history, user_clicks, final_results, send_btn_update, *textbox_update_list, *click_bttn_list

# Gradio界面

if __name__ == "__main__":
    
    # load inference engine list
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_retrieval_config_path", type=str, required=True, help="Retrieval inference config file")  
    parser.add_argument("--infer_ranker_config_path", type=str, required=True, help="Ranker inference config file")  
    args = parser.parse_args()
    with open(args.infer_retrieval_config_path, 'r') as f:
        infer_retrieval_config = yaml.safe_load(f)
    with open(args.infer_ranker_config_path, 'r') as f:
        infer_ranker_config = yaml.safe_load(f)
    retrieval_inference_engine = RetrieverInferenceEngine(infer_retrieval_config)
    rank_inference_engine = RankerInferenceEngine(infer_ranker_config)
    inference_engine_list:list[InferenceEngine] = [retrieval_inference_engine, rank_inference_engine]
    final_topk = inference_engine_list[-1].config['output_topk']
    
    # get all keys from redis
    unique_keys = []
    user_id2latest_timestamp = {}
    keys = list(inference_engine_list[-1].redis_client.scan_iter(match='recflow:user_timestamp*', count=10000))
    for key in keys:
        user_id, timestamp = key.decode().split(':')[-1].split('_')  # 分离 user_id 和 timestamp
        if user_id not in user_id2latest_timestamp:
            unique_keys.append(int(user_id))
            user_id2latest_timestamp[int(user_id)] = timestamp
        else:
            if timestamp is None or timestamp > user_id2latest_timestamp[user_id]:
                user_id2latest_timestamp[user_id] = timestamp
    unique_keys = np.random.choice(unique_keys, size=min(len(unique_keys), 100), replace=False)

    # initialize show_items
    init_result = [{"id": -1, "name": None} for i in range(final_topk)]

    with gr.Blocks() as demo:
        gr.Markdown("## Multistage Recommendation System")
        show_items = gr.State(init_result)

        with gr.Row():  # 最外层是一个 Row
            with gr.Column(scale=4):  # 第一列是 Chatbot
                gr.Markdown("### Inner Pipeline")
                user_id = gr.Dropdown(label="User ID", value=None, choices=unique_keys.tolist())
                chatbot = gr.Chatbot(type="messages",
                                     value=[{"role": "assistant", 
                                             "content": "🤗 Hello, this is a multistage recommendation system, please choose a user."}])
                user_clicks = gr.Textbox(label="Click history", interactive=False)
                send_btn = gr.Button("Request", interactive=False)
            
            item_textboxes = []
            item_buttons = []
            with gr.Column(scale=4):  # 第二列是商品列表
                gr.Markdown("### Item List")
                for i in range(final_topk):
                    with gr.Row():
                        textbox = gr.Textbox(value=f"-", visible=False, container=False, scale=6)
                        button = gr.Button(value=f"Click", elem_id=f"btn_{i}", visible=False, scale=2)
                        button.click(fn=button_click_fns(i), inputs=[user_clicks, show_items], outputs=[user_clicks])
                        item_textboxes.append(textbox)
                        item_buttons.append(button)

            # Dropdown 的选择事件绑定回调
            user_id.change(fn=change_user_fns, inputs=user_id, 
                           outputs=[chatbot, user_clicks, show_items, send_btn, *item_textboxes, *item_buttons])
            # 发送按钮点击事件，触发chat_interface
            send_btn.click(fn=chat_interface, inputs=[user_clicks, chatbot, user_id], 
                           outputs=[chatbot, user_clicks, show_items, *item_textboxes, *item_buttons])

    # 启动Gradio界面
    demo.launch()
