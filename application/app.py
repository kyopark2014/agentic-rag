import streamlit as st 
import chat

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
    debugMode = st.selectbox(
        '🖊️ 디버그 모드를 설정하세요',
        ('Debug', 'Normal')
    )

    st.success("Connected to Nova Pro", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # print('clear_button: ', clear_button)

st.title('🔮 '+ mode)

if clear_button==True:
    chat.initiate()

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
            st.rerun()

            chat.save_chat_history(prompt, response)

        elif mode == 'RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_rag_with_knowledge_base(prompt, st, debugMode)        
                st.write(response)
                print('response: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Debug":
                    st.rerun()

                chat.save_chat_history(prompt, response)

        elif mode == 'Agentic RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_agent_executor(prompt, st, debugMode)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Debug":
                    st.rerun()

                chat.save_chat_history(prompt, response)
        
        elif mode == 'Corrective RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_basic_rag(prompt, st, debugMode)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Debug":
                    st.rerun()

                chat.save_chat_history(prompt, response)

        elif mode == 'Self RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_planning(prompt, st, debugMode)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Debug":
                    st.rerun()

                chat.save_chat_history(prompt, response)

        elif mode == 'Self Corrective RAG':
            with st.status("thinking...", expanded=True, state="running") as status:
                response = chat.run_long_form_writing_agent(prompt, st, debugMode)
                st.write(response)
                print('response: ', response)

                if response.find('<thinking>') != -1:
                    print('Remove <thinking> tag.')
                    response = response[response.find('</thinking>')+12:]
                    print('response without tag: ', response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                if debugMode != "Debug":
                    st.rerun()

                chat.save_chat_history(prompt, response)

        else:
            stream = chat.general_conversation(prompt)

            response = st.write_stream(stream)
            print('response: ', response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)
        


