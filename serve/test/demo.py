# 运行
# pip install gradio
# python demo.py


import gradio as gr
import time


N = 10  # 展示的商品数量


# 模拟多阶段检索过程
def retrieve(user_id, user_clicks):
    """
    Args:
        user_id (int): 用户ID
        user_clicks (str): 用户点击的商品ID字符串, 以空格分隔

    Returns:
        results (List[str]): 每个阶段的检索结果
        final_result (List[Dict]): 最终展示的结果
    """
    # 模拟一些检索阶段
    stages = [
        f"Stage 1: Initial data retrieval for user {user_id}",
        "Stage 2: Data filtering and processing",
        "Stage 3: Generating final results"
    ]
    
    results = []
    
    # 模拟每个阶段的检索过程
    for stage in stages:
        time.sleep(1)  # 模拟延时
        results.append(stage)
    
    # 最终结果
    final_result = [
        {"id": i, "name": "Product A", "image_path": "https://imgslim.geekpark.net/uploads/image/file/12/8b/128b73603f95d9310bde55db9daa1e35.jpeg"}
        for i in range(100, 100+N)
    ]
    return results, final_result


def chat_interface(user_clicks, chat_history, user_id):
    # request 按钮触发的动作
    # 获取多阶段检索的中间结果和最终结果（返回一组ID）
    stages_results, final_results = retrieve(user_id, user_clicks)
    
    # 为每个阶段添加折叠展示
    stage_messages = []
    for i, result in enumerate(stages_results):
        stage_messages.append(f"<details><summary>Stage {i+1}</summary><p>{result}</p></details>")
    
    # 更新对话历史
    chat_history.append({"role": "user", "content": "User click: "+user_clicks})
    chat_history.append({"role": "assistant", "content": "\n".join(stage_messages)})
    
    # 更新展示内容
    textbox_list = [f"item-{item['id']}: {item['name']}" for item in final_results]

    user_clicks = ""    # 每次请求，清空上次的点击历史。因为默认历史已入库保留

    return chat_history, user_clicks, final_results, *textbox_list  # 返回聊天历史和按钮内容

# 处理用户点击的按钮
def button_click_fns(i):
    def fn(user_history, show_items):
        # 模拟用户点击某个按钮的ID，收集这个ID作为新的用户输入
        user_history = " ".join([user_history, str(show_items[int(i)]["id"])]).strip()
        return user_history
    return fn

# Gradio界面

# 初始展示结果
init_result = [
    {"id": i, "name": "Product B", "image_path": "https://imgslim.geekpark.net/uploads/image/file/12/8b/128b73603f95d9310bde55db9daa1e35.jpeg"}
    for i in range(N)
]


with gr.Blocks() as demo:
    gr.Markdown("## Multistage Recommendation System Demo")
    show_items = gr.State(init_result)

    with gr.Row():  # 最外层是一个 Row
        with gr.Column(scale=4):  # 第一列是 Chatbot
            gr.Markdown("### Inner Pipeline")
            user_id = gr.Dropdown(label="User ID", value=0, choices=list(range(10)))
            chatbot = gr.Chatbot(type="messages")
            user_clicks = gr.Textbox(label="Click history", interactive=False)
            send_btn = gr.Button("Request")
        
        item_textboxes = []
        item_buttons = []
        with gr.Column(scale=8):  # 第二列是商品列表
            gr.Markdown("### Item List")
            for i in range(N):
                with gr.Row():
                    textbox = gr.Textbox(value=f"item-{init_result[i]['id']}: {init_result[i]['name']}", visible=True, container=False, scale=6)
                    button = gr.Button(value=f"Click", elem_id=f"btn_{i}", scale=2)
                    button.click(fn=button_click_fns(i), inputs=[user_clicks, show_items], outputs=[user_clicks])
                    item_textboxes.append(textbox)
                    item_buttons.append(button)

        # 发送按钮点击事件，触发chat_interface
        send_btn.click(fn=chat_interface, inputs=[user_clicks, chatbot, user_id], outputs=[chatbot, user_clicks, show_items, *item_textboxes])

# 启动Gradio界面
demo.launch()
