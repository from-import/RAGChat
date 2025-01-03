import autogen
from rich import print
import chainlit as cl
from typing_extensions import Annotated
from chainlit.input_widget import (
    Select, Slider, Switch)
from autogen import AssistantAgent, UserProxyAgent
from utils.chainlit_agents import ChainlitUserProxyAgent, ChainlitAssistantAgent
from graphrag.query.cli import run_global_search, run_local_search

# 配置LLM参数(示例中使用 Lite-LLM Server)：
llm_config_autogen = {
    "seed": 42,  # 随机种子，可调整以产生不同实验结果
    "temperature": 0,  # 温度，控制生成文本的随机性程度
    "config_list": [
        {
            "model": "litellm",
            "base_url": "http://0.0.0.0:4000/",  # Lite-LLM Server地址
            'api_key': 'ollama'
        },
    ],
    "timeout": 60000,  # 超时时间，单位毫秒
}


@cl.on_chat_start
async def on_chat_start():
    """
    当聊天开始时 (第一次打开或刷新时) 执行该函数，用于初始化界面和会话参数。
    """
    try:
        # 1. 创建一个 ChatSettings，让用户在前端勾选或设置参数
        #    ChatSettings 传入一个列表，列表中每个元素都会在前端显示相应组件
        #    Switch/Select/Slider等都是 chainlit 提供的组件
        settings = await cl.ChatSettings(
            [
                # 开关按钮，用于切换本地搜索或全局搜索
                Switch(id="Search_type", label="(GraphRAG) Local Search", initial=True),

                # 下拉选择，用于选择生成内容的类型
                Select(
                    id="Gen_type",
                    label="(GraphRAG) Content Type",
                    values=["prioritized list", "single paragraph", "multiple paragraphs", "multiple-page report"],
                    initial_index=1,  # 初始值设置为“single paragraph”
                ),

                # 滑动条，用于设置“Community Level”
                Slider(
                    id="Community",
                    label="(GraphRAG) Community Level",
                    initial=0,
                    min=0,
                    max=2,
                    step=1,
                ),
            ]
        ).send()

        # 2. 从用户界面获取返回设置
        response_type = settings["Gen_type"]
        community = settings["Community"]
        local_search = settings["Search_type"]

        # 3. 将用户的配置项保存在会话变量中
        cl.user_session.set("Gen_type", response_type)
        cl.user_session.set("Community", community)
        cl.user_session.set("Search_type", local_search)

        # 4. 定义一个“Retriever”代理，用于后续搜集/检索上下文等
        retriever = AssistantAgent(
            name="Retriever",
            llm_config=llm_config_autogen,
            system_message="""Only execute the function query_graphRAG to look for context. 
                    Output 'TERMINATE' when an answer has been provided.""",
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",  # 该代理不会主动需要人工输入
            description="Retriever Agent"
        )

        # 5. 定义一个“User Proxy”代理，用于模拟用户，对接Retriever
        user_proxy = ChainlitUserProxyAgent(
            name="User_Proxy",
            human_input_mode="ALWAYS",  # 该代理会从用户输入中获取消息
            llm_config=llm_config_autogen,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
            system_message='''A human admin. Interact with the retriever to provide any context''',
            description="User Proxy Agent"
        )

        print("Set agents.")

        # 6. 将这两个代理对象存储到会话中，方便后续使用
        cl.user_session.set("Query Agent", user_proxy)
        cl.user_session.set("Retriever", retriever)

        # 7. 向前端发送一条消息，询问用户想要执行的任务
        msg = cl.Message(
            content="""Hello! What task would you like to get done today?""",
            author="User_Proxy"
        )
        await msg.send()

        print("Message sent.")

    except Exception as e:
        # 如果执行过程中出现错误，则打印错误信息
        print("Error: ", e)
        pass


@cl.on_settings_update
async def setup_agent(settings):
    """
    当用户通过界面更改设置时（例如切换 Search_type、修改 Gen_type、拖动 Slider 等），
    该函数会被触发，并更新会话中对应的设置变量。
    """
    response_type = settings["Gen_type"]
    community = settings["Community"]
    local_search = settings["Search_type"]

    # 更新会话变量
    cl.user_session.set("Gen_type", response_type)
    cl.user_session.set("Community", community)
    cl.user_session.set("Search_type", local_search)

    print("on_settings_update", settings)


@cl.on_message
async def run_conversation(message: cl.Message):
    """
    当用户在前端输入消息并提交时，会调用该回调函数。
    """
    print("Running conversation")

    # 一些上下文参数可在此修改
    INPUT_DIR = None  # 文件输入目录(示例中为 None)
    ROOT_DIR = '.'  # 搜索功能可能用到的根目录
    CONTEXT = message.content  # 用户输入的内容
    MAX_ITER = 10  # 设置一次对话的最大轮数

    # 从会话中读取先前存储的参数
    RESPONSE_TYPE = cl.user_session.get("Gen_type")
    COMMUNITY = cl.user_session.get("Community")
    LOCAL_SEARCH = cl.user_session.get("Search_type")

    # 读取预先保存的代理对象
    retriever = cl.user_session.get("Retriever")
    user_proxy = cl.user_session.get("Query Agent")

    print("Setting groupchat")

    # 定义一个函数，用于决定下一次说话的人是谁(代理之间的对话)
    def state_transition(last_speaker, groupchat):
        """
        根据当前对话的最后发言者，决定下一步由哪个代理回答。
        如果最后说话的是 user_proxy -> 交给 retriever。
        如果最后说话的是 retriever -> 再返回给 user_proxy。
        如有其他需求，可自行拓展。
        """
        messages = groupchat.messages
        if last_speaker is user_proxy:
            return retriever
        if last_speaker is retriever:
            # 简单逻辑示例：如果 retriever 最后说的话并不是 "math_expert" 或 "physics_expert"，
            # 就继续返回 user_proxy。如果是其中一个，则也返回 user_proxy（示例逻辑）。
            if messages[-1]["content"].lower() not in ['math_expert', 'physics_expert']:
                return user_proxy
            else:
                if messages[-1]["content"].lower() == 'math_expert':
                    return user_proxy
                else:
                    return user_proxy
        else:
            # 默认返回None，通常不会执行到
            return None

    # 定义一个在retriever代理中被调用的函数，用于执行GraphRAG搜索
    async def query_graphRAG(
            question: Annotated[str, 'Query string containing information that you want from RAG search']
    ) -> str:
        """
        当 retriever 代理需要执行“查询GraphRAG”的功能时，会调用这个函数。
        根据用户设置的 Search_type 决定是本地搜索或全局搜索，并将结果发送给前端显示。
        """
        if LOCAL_SEARCH:
            print(LOCAL_SEARCH)  # 打印当前搜索类型(仅用于调试)
            result = run_local_search(INPUT_DIR, ROOT_DIR, COMMUNITY, RESPONSE_TYPE, question)
        else:
            result = run_global_search(INPUT_DIR, ROOT_DIR, COMMUNITY, RESPONSE_TYPE, question)

        # 将搜索结果以消息形式发送到前端
        await cl.Message(content=result).send()
        return result

    # 将 query_graphRAG 函数注册到 retriever 中，用于描述retriever可执行的方法
    for caller in [retriever]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for code generation and question answering.",
            api_style="function"
        )(query_graphRAG)

    # 将上面定义的 d_retrieve_content 函数注册给 user_proxy 和 retriever，以便它们都可以使用
    for agents in [user_proxy, retriever]:
        agents.register_for_execution()(d_retrieve_content)

    # 创建一个 groupchat，包含 user_proxy 和 retriever 两个代理，设置对话策略
    groupchat = autogen.GroupChat(
        agents=[user_proxy, retriever],
        messages=[],
        max_round=MAX_ITER,  # 最大轮数
        speaker_selection_method=state_transition,  # 使用上面定义的 state_transition 函数
        allow_repeat_speaker=True,  # 允许同一个代理连续说话
    )

    # 创建一个 groupchat 管理器，用于管理对话流
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config_autogen,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
    )

    # 下面是对话的主要流程控制：
    if len(groupchat.messages) == 0:
        # 如果当前对话消息数是0，表示是初次对话。
        # 则 user_proxy 代理以 CONTEXT(用户输入) 作为第一条消息进行对话初始化
        await cl.make_async(user_proxy.initiate_chat)(
            manager,
            message=CONTEXT
        )
    elif len(groupchat.messages) < MAX_ITER:
        # 如果还未达到最大对话轮数，则继续发送用户的消息
        await cl.make_async(user_proxy.send)(
            manager,
            message=CONTEXT
        )
    elif len(groupchat.messages) == MAX_ITER:
        # 如果达到了最大对话轮数，则发送退出消息
        await cl.make_async(user_proxy.send)(
            manager,
            message="exit"
        )
