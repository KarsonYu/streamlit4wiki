import streamlit as st
from langchain_zhipu import ChatZhipuAI
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
from dotenv import load_dotenv, find_dotenv


# Streamlit 应用程序界面
def main():
    st.title('动手构建知识库')
    api_key = st.sidebar.text_input('API Key', type='password')

    # 用于跟踪对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 检查API Key是否已填写
    if not api_key:
        st.warning("⚠️ 请在侧边栏填写API Key后继续")
        return
    
    # 存储检索问答链
    try:
        if "qa_history_chain" not in st.session_state or st.session_state.get("current_api_key") != api_key:
            st.session_state.qa_history_chain = get_qa_history_chain(api_key)
            st.session_state.current_api_key = api_key
    except Exception as e:
        st.error(f"⚠️ 初始化模型失败: {str(e)}")
        return
    
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages:
            with messages.chat_message(message[0]):
                st.write(message[1])
    
    if prompt := st.chat_input("请输入您的问题..."):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        with messages.chat_message("human"):
            st.write(prompt)

        try:
            answer = gen_response(
                chain=st.session_state.qa_history_chain,
                input=prompt,
                chat_history=st.session_state.messages
            )
            with messages.chat_message("ai"):
                output = st.write_stream(answer)
            st.session_state.messages.append(("ai", output))
        except Exception as e:
            error_message = f"⚠️ 模型调用失败: {str(e)}"
            with messages.chat_message("ai"):
                st.error(error_message)
            st.session_state.messages.append(("ai", error_message)) 

def get_retriever(api_key=None):
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings(api_key=api_key)
    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain(api_key):
    retriever = get_retriever(api_key=api_key)
    llm = ChatZhipuAI(temperature=0.2, api_key=api_key)
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain

def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]



if __name__ == "__main__":
    # _ = load_dotenv(find_dotenv())
    main()