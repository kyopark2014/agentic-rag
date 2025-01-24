import streamlit as st 
import chat
import time
import uuid 

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "RAG": [
        "기본적인 RAG로 Hallucination을 최소화하고 애플리케이션에 필요한 정보를 제공합니다."
    ],
    "Agentic RAG": [
        "Agent를 이용해 RAG의 성능을 향상시킵니다."
    ],
    "Corrective RAG": [
        "Corrective RAG를 활용하여 RAG의 성능을 향상 시킵니다."
    ],
    "Self RAG": [
        "Self RAG를 활용하여 RAG의 성능을 향상 시킵니다."
    ],
    "Self Corrective RAG": [
        "Self Corrective RAG를 활용하여 RAG의 성능을 향상 시킵니다."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Amazon Bedrock을 이용해 다양한 형태의 대화를 구현합니다." 
        "여기에서는 일상적인 대화와 각종 툴을 이용해 Agent를 구현할 수 있습니다." 
        "또한 번역이나 문법 확인과 같은 용도로 사용할 수 있습니다."
        "주요 코드는 LangChain과 LangGraph를 이용해 구현되었습니다.\n"
        "상세한 코드는 [Github](https://github.com/kyopark2014/agentic-rag)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "RAG", "Agentic RAG", "Corrective RAG", "Self RAG", "Self Corrective RAG"], index=0
    )   
    st.info(mode_descriptions[mode][0])    
    # print('mode: ', mode)

    # debug Mode
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ('Nova Pro', 'Nova Lite', 'Claude Sonnet 3.5', 'Claude Sonnet 3.0', 'Claude Haiku 3.5')
    )
    
    # debug Mode
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    #print('debugMode: ', debugMode)

    # debug Mode
    select_multiRegion = st.checkbox('Multi Region', value=True)
    multiRegion = 'Enable' if select_multiRegion else 'Disable'
    #print('multiRegion: ', multiRegion)

    # contextual embedding
    selected_contextualEmbedding = st.checkbox('Contextual Embedding', value=False)
    contextualEmbedding = 'Enable' if selected_contextualEmbedding else 'Disable'
    #print('contextualEmbedding: ', contextualEmbedding)

    chat.update(modelName, debugMode, multiRegion, contextualEmbedding)

    st.subheader("📋 문서 업로드")
    print('fileId: ', chat.fileId)
    uploaded_file = st.file_uploader("RAG를 위한 파일을 선택합니다.", type=["pdf", "doc", "docx", "ppt", "pptx", "png", "jpg", "jpeg", "txt", "py", "md", "csv"], key=chat.fileId)

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # print('clear_button: ', clear_button)

st.title('🔮 '+ mode)  

if clear_button==True:
    chat.initiate()

# Preview the uploaded image in the sidebar
file_name = ""
if uploaded_file is not None and clear_button==False:
    if uploaded_file.name:      
        chat.initiate()

        if debugMode=='Enable':
            status = '이미지를 업로드합니다.'
            print('status: ', status)
            st.info(status)

        file_name = uploaded_file.name
        file_url = chat.upload_to_s3(uploaded_file.getvalue(), file_name, contextualEmbedding)
        print('file_url: ', file_url) 
            
        progress_text = f'선택한 "{file_name}"을 업로드하고 파일 내용을 요약하고 있습니다...'
        # my_bar = st.sidebar.progress(0, text=progress_text)
        
        # for percent_complete in range(100):
        #     time.sleep(0.2)
        #     my_bar.progress(percent_complete + 1, text=progress_text)
        if debugMode=='Enable':
            print('status: ', progress_text)
            st.info(progress_text)
    
        msg = chat.get_summary_of_uploaded_file(file_name, st)
        st.session_state.messages.append({"role": "assistant", "content": f"선택한 문서({file_name})를 요약하면 아래와 같습니다.\n\n{msg}"})    
        print('msg: ', msg)
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

display_chat_messages()

def show_references(reference_docs):
    if debugMode == "Enable" and reference_docs:
        with st.expander(f"답변에서 참조한 {len(reference_docs)}개의 문서입니다."):
            for i, doc in enumerate(reference_docs):
                st.markdown(f"**{doc.metadata['name']}**: {doc.page_content}")
                st.markdown("---")

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    uploaded_file = None
    
    st.session_state.greetings = False
    st.rerun()

    chat.clear_chat_history()

if mode == 'Agentic RAG':
    col1, col2, col3 = st.columns([0.1, 0.25, 0.1])
    url = "https://raw.githubusercontent.com/kyopark2014/agentic-rag/main/contents/agentic-rag.png"
    col2.image(url)
elif mode == 'Corrective RAG':
    col1, col2, col3 = st.columns([0.2, 0.3, 0.2])
    url = "https://raw.githubusercontent.com/kyopark2014/agentic-rag/main/contents/corrective-rag.png"
    col2.image(url)    
elif mode == 'Self RAG':
    col1, col2, col3 = st.columns([0.1, 2.0, 0.1])
    url = "https://raw.githubusercontent.com/kyopark2014/agentic-rag/main/contents/self-rag2.png"
    col2.image(url)
elif mode == 'Self Corrective RAG':
    col1, col2, col3 = st.columns([0.1, 2.0, 0.1])    
    url = "https://raw.githubusercontent.com/kyopark2014/agentic-rag/main/contents/self-corrective-rag.png"
    col2.image(url)

# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    print('prompt: ', prompt)

    with st.chat_message("assistant"):
        if mode == '일상적인 대화':
            stream = chat.general_conversation(prompt)
            response = st.write_stream(stream)
            print('response: ', response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)

        elif mode == 'RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = chat.get_answer_using_opensearch(prompt, st)     
                st.write(response)
                print('response: ', response)                  
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Agentic RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_agent_executor(prompt, st)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 
                        
        elif mode == 'Corrective RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_corrective_rag(prompt, st)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Self RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_self_rag(prompt, st)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)

            show_references(reference_docs) 

        elif mode == 'Self Corrective RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_self_corrective_rag(prompt, st)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Enable":
                    st.rerun()

                chat.save_chat_history(prompt, response)

            show_references(reference_docs) 

        else:
            stream = chat.general_conversation(prompt)

            response = st.write_stream(stream)
            print('response: ', response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)
        
