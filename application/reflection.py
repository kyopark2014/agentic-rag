import utils
import chat
import traceback
import rag_opensearch as rag
import tool_use
import search

from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from typing_extensions import Annotated, TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

logger = utils.CreateLogger("reflection")

reference_docs = []

useEnhancedSearch = False

####################### LangGraph #######################
# Agentic Workflow: Reflection
#########################################################
def extract_reflection(draft):
    class Reflection(BaseModel):
        missing: str = Field(description="Critique of what is missing.")
        advisable: str = Field(description="Critique of what is helpful for better answer")
        superfluous: str = Field(description="Critique of what is superfluous")

    class Research(BaseModel):
        """Provide reflection and then follow up with search queries to improve the answer."""

        reflection: Reflection = Field(description="Your reflection on the initial answer.")
        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current answer."
        )
    
    class ReflectionKor(BaseModel):
        missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
        advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
        superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

    class ResearchKor(BaseModel):
        """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""

        reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
        search_queries: list[str] = Field(
            description="도출된 비평을 해결하기 위한 3개 이내의 검색어"            
        )    

    reflection = []
    search_queries = []
    for attempt in range(5):
        try:
            llm = chat.get_chat(extended_thinking="Disable")
            if chat.isKorean(draft):
                structured_llm = llm.with_structured_output(Research, include_raw=True)
            else:
                structured_llm = llm.with_structured_output(Research, include_raw=True)
            
            info = structured_llm.invoke(draft)
            logger.info(f"attempt: {attempt}, info: {info}")
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                logger.info(f"parsed_info: {parsed_info}")
                reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                search_queries = parsed_info.search_queries
                
                logger.info(f"reflection: { parsed_info.reflection}")
                logger.info(f"search_queries: {search_queries}")

        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}") 

        return reflection, search_queries

def extract_reflection2(draft):
    system = (
        "주어진 문장을 향상시키기 위하여 아래와 같은 항목으로 개선사항을 추출합니다."
        "missing: 작성된 글에 있어야하는데 빠진 내용이나 단점"
        "advisable: 더 좋은 글이 되기 위해 추가하여야 할 내용"
        "superfluous: 글의 길이나 스타일에 대한 비평"    
        "<result> tag를 붙여주세요."
    )
    critique_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{draft}"),
        ]
    )

    reflection = ""
    for attempt in range(5):
        try:
            llm = chat.get_chat(extended_thinking="Disable")
            chain = critique_prompt | llm
            result = chain.invoke({
                "draft": draft
            })
            logger.info(f"result: {result}")

            output = result.content

            if output.find('<result>') != -1:
                output = output[output.find('<result>')+8:output.find('</result>')]
            logger.info(f"output: {output}")

            reflection = output            
            break
                
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}") 

    # search queries
    search_queries = []
    class Queries(BaseModel):
        """Provide reflection and then follow up with search queries to improve the answer."""

        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current answer."
        )
    class QueriesKor(BaseModel):
        """글쓰기를 개선하기 위한 검색어를 제공합니다."""

        search_queries: list[str] = Field(
            description="주어진 비평을 해결하기 위한 3개 이내의 검색어"            
        )    

    system = (
        "당신은 주어진 Draft를 개선하여 더 좋은 글쓰기를 하고자 합니다."
        "주어진 비평을 반영하여 초안을 개선하기 위한 3개 이내의 검색어를 추천합니다."
    )
    queries_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Draft: {draft} \n\n Critiques: {reflection}"),
        ]
    )
    for attempt in range(5):
        try:
            llm = chat.get_chat(extended_thinking="Disable")
            if chat.isKorean(draft):
                structured_llm_queries = llm.with_structured_output(QueriesKor, include_raw=True)
            else:
                structured_llm_queries = llm.with_structured_output(Queries, include_raw=True)

            retrieval_quries = queries_prompt | structured_llm_queries
            
            info = retrieval_quries.invoke({
                "draft": draft,
                "reflection": reflection
            })
            logger.info(f"attempt: {attempt}, info: {info}")
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                logger.info(f"parsed_info: {parsed_info}")
                search_queries = parsed_info.search_queries
                logger.info(f"search_queries: {search_queries}")
            break
                
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}") 

    return reflection, search_queries

def run_reflection(query, st):
    class State(TypedDict):
        task: str
        draft: str
        reflection: list
        search_queries: list
        revision_number: int
            
    def generate(state: State, config):    
        logger.info(f"###### generate ######")
        logger.info(f"task: {state['task']}")

        global reference_docs

        query = state['task']

        # grade   
        if chat.debug_mode == "Enable":
            st.info(f"초안(draft)를 생성하기 위하여, 정보를 조회합니다.") 

        top_k = 4
        relevant_docs = rag.retrieve_documents_from_opensearch(query, top_k=top_k)  
        if chat.internet_mode == "Enable":
            relevant_docs += search.retrieve_documents_from_tavily(query, top_k=top_k)
    
        # grade   
        if chat.debug_mode == "Enable":
            st.info(f"가져온 {len(relevant_docs)}개의 문서를 평가하고 있습니다.") 

        filtered_docs = chat.grade_documents(query, relevant_docs)    
        filtered_docs = chat.check_duplication(filtered_docs) # duplication checker
        if len(filtered_docs):
            reference_docs += filtered_docs 

        if chat.debug_mode == "Enable":
            st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")
        
        # generate
        if chat.debug_mode == "Enable":
            st.info(f"초안을 생성중입니다.")
        
        relevant_context = ""
        for document in filtered_docs:
            relevant_context = relevant_context + document.page_content + "\n\n"        
        # print('relevant_context: ', relevant_context)

        rag_chain = chat.get_rag_prompt(query)
                        
        draft = ""    
        try: 
            result = rag_chain.invoke(
                {
                    "question": query,
                    "context": relevant_context                
                }
            )
            logger.info(f"result: {result}")

            draft = result.content        
            if draft.find('<result>')!=-1:
                draft = draft[draft.find('<result>')+8:draft.find('</result>')]
            
            if chat.debug_mode=="Enable":
                st.info(f"생성된 초안: {draft}")
            
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            raise Exception ("Not able to request to LLM")
        
        return {"draft":draft}
    
    def reflect(state: State, config):
        logger.info(f"###### reflect ######")
        logger.info(f"draft: {state['draft']}")

        draft = state["draft"]
        
        if chat.debug_mode=="Enable":
            st.info('초안을 검토하여 부족하거나 보강할 내용을 찾고, 추가 검색어를 추출합니다.')

        reflection, search_queries = extract_reflection2(draft)
        if chat.debug_mode=="Enable":  
            st.info(f'개선할 사항: {reflection}')
            st.info(f'추가 검색어: {search_queries}')        

        return {
            "reflection": reflection,
            "search_queries": search_queries
        }
    
    def get_revise_prompt(text):
        llm = chat.get_chat(extended_thinking="Disable")

        if chat.isKorean(text):
            system = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                "draft를 critique과 information 참조하여 수정하세오."
                "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
            )
            human = (
                "draft:"
                "{draft}"
                            
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )
        else:
            system = (
                "You are an excellent writing assistant." 
                "Revise this draft using the critique and additional information."
                "Provide the final answer with <result> tag."
            )
            human = (
                "draft:"
                "{draft}"
                            
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )                    
        revise_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system),
                ("human", human),
            ]
        )                    
        revise_chain = revise_prompt | llm

        return revise_chain
    
    def revise_answer(state: State, config):           
        logger.info(f"###### revise_answer ######")

        if chat.debug_mode=="Enable":
            st.info("개선할 사항을 반영하여 답변을 생성중입니다.")
        
        top_k = 2        
        selected_docs = []
        for q in state["search_queries"]:
            relevant_docs = []
            filtered_docs = []
            if chat.debug_mode=="Enable":
                st.info(f"검색을 수행합니다. 검색어: {q}")
        
            relevant_docs = rag.retrieve_documents_from_opensearch(q, top_k)     

            if chat.internet_mode=="Enable":
                relevant_docs += search.retrieve_documents_from_tavily(q, top_k)

            # grade   
            if chat.debug_mode == "Enable":
                st.info(f"가져온 {len(relevant_docs)}개의 문서를 평가하고 있습니다.") 

            filtered_docs += chat.grade_documents(q, relevant_docs) # grading    

            if chat.debug_mode == "Enable":
                st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")

            selected_docs += filtered_docs

        selected_docs += chat.check_duplication(selected_docs) # check duplication
        
        global reference_docs
        if relevant_docs:
            reference_docs += selected_docs

        if chat.debug_mode == "Enable":
            st.info(f"최종으로 {len(reference_docs)}개의 문서가 선택되었습니다.")

        content = ""
        if len(relevant_docs):
            for d in relevant_docs:
                content += d.page_content+'\n\n'
            logger.info(f"content: {content}")

        for attempt in range(5):
            logger.info(f"attempt: {attempt}")

            revise_chain = get_revise_prompt(state['task'])
            try:
                res = revise_chain.invoke(
                    {
                        "draft": state['draft'],
                        "reflection": state["reflection"],
                        "content": content
                    }
                )
                output = res.content
                logger.info(f"output: {output}")

                if output.find('<result>')==-1:
                    draft = output
                else:
                    draft = output[output.find('<result>')+8:output.find('</result>')]

                logger.info(f"revised_answer: {draft}")
                break

            except Exception:
                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}")
                
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "draft": draft, 
            "revision_number": revision_number + 1
        }
    
    MAX_REVISIONS = 1
    def should_continue(state: State, config):
        logger.info(f"###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        logger.info(f"max_revisions: {max_revisions}")
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"

    def buildReflection():    
        workflow = StateGraph(State)

        workflow.add_node("generate", generate)
        workflow.add_node("reflect", reflect)
        workflow.add_node("revise_answer", revise_answer)

        workflow.set_entry_point("generate")

        workflow.add_conditional_edges(
            "revise_answer", 
            should_continue, 
            {
                "end": END, 
                "continue": "reflect"}
        )

        workflow.add_edge("generate", "reflect")
        workflow.add_edge("reflect", "revise_answer")
        
        app = workflow.compile()
        
        return app
    
    # initiate
    global contentList, reference_docs
    contentList = []
    reference_docs = []

    # workflow
    app = buildReflection()
        
    inputs = {
        "task": query
    } 
    config = {
        "recursion_limit": 50,
        "max_revisions": MAX_REVISIONS,
        "parallel_processing": chat.multi_region
    }
    
    output = app.invoke(inputs, config)
    logger.info(f"output: {output}")
        
    msg = output["draft"]

    reference = ""
    if reference_docs:
        reference = chat.get_references(reference_docs)

    return msg+reference, reference_docs

####################### LangGraph #######################
# Agentic Workflow: Reflection (run_knowledge_guru)
#########################################################

def init_enhanced_search(st):
    llm = chat.get_chat(extended_thinking="Disable") 

    model = llm.bind_tools(tool_use.tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tool_node = ToolNode(tool_use.tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
            
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:                
            return "continue"

    def call_model(state: State, config):
        logger.info(f"##### call_model #####")

        messages = state["messages"]
        # print('messages: ', messages)

        last_message = messages[-1]
        logger.info(f"last_message: {last_message}")

        if isinstance(last_message, ToolMessage) and last_message.content=="":              
            logger.info(f"last_message is empty")
            logger.info(f"question: {state['messages'][0].content}")
            answer = chat.get_basic_answer(state['messages'][0].content)          
            return {"messages": [AIMessage(content=answer)]}
            
        if chat.isKorean(messages[0].content)==True:
            system = (
                "당신은 질문에 답변하기 위한 정보를 수집하는 연구원입니다."
                "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "최종 답변에는 조사한 내용을 반드시 포함하여야 하고, <result> tag를 붙여주세요."
            )
        else: 
            system = (            
                "You are a researcher charged with providing information that can be used when making answer."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will be acting as a thoughtful advisor."
                "Put it in <result> tags."
            )
                
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt | model
                
        response = chain.invoke(messages)
        logger.info(f"call_model response: {response}")
              
        # state messag
        if response.tool_calls:
            logger.info(f"tool_calls response: {response.tool_calls}")

            toolinfo = response.tool_calls[-1]            
            if toolinfo['type'] == 'tool_call':
                logger.info(f"tool name: {toolinfo['name']}")    

            if chat.debug_mode=="Enable":
                st.info(f"{response.tool_calls[-1]['name']}: {response.tool_calls[-1]['args']}")
                   
        return {"messages": [response]}

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
            
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")
        return workflow.compile()
    
    return buildChatAgent()

def enhanced_search(query, config, st):
    logger.info(f"###### enhanced_search ######")
    inputs = [HumanMessage(content=query)]

    app_enhanced_search = init_enhanced_search(st)        
    result = app_enhanced_search.invoke({"messages": inputs}, config)   
    logger.info(f"result: {result}")
            
    message = result["messages"][-1]
    logger.info(f"enhanced_search: {message}")

    if message.content.find('<result>')==-1:
        return message.content
    else:
        return message.content[message.content.find('<result>')+8:message.content.find('</result>')]
    
def run_knowledge_guru(query, st):
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        reflection: list
        search_queries: list
            
    def generate(state: State, config):    
        logger.info(f"###### generate ######")
        logger.info(f"state: {state['messages']}")
        logger.info(f"task: {state['messages'][0].content}")

        if chat.debug_mode=="Enable":
            st.info(f"검색을 수행합니다. 검색어: {state['messages'][0].content}")
        
        draft = enhanced_search(state['messages'][0].content, config, st)  
        logger.info(f"draft: {draft}")

        if chat.debug_mode=="Enable":
            st.info(f"생성된 초안: {draft}")
        
        return {
            "messages": [AIMessage(content=draft)]
        }
    
    class Reflection(BaseModel):
        missing: str = Field(description="Critique of what is missing.")
        advisable: str = Field(description="Critique of what is helpful for better answer")
        superfluous: str = Field(description="Critique of what is superfluous")

    class Research(BaseModel):
        """Provide reflection and then follow up with search queries to improve the answer."""

        reflection: Reflection = Field(description="Your reflection on the initial answer.")
        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current answer."
        )
    
    class ReflectionKor(BaseModel):
        missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
        advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
        superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

    class ResearchKor(BaseModel):
        """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""

        reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
        search_queries: list[str] = Field(
            description="도출된 비평을 해결하기 위한 3개 이내의 검색어"            
        )    

    def reflect(state: State, config):
        logger.info(f"###### reflect ######")
        logger.info(f"state: {state['messages']}")
        logger.info(f"draft: {state['messages'][-1].content}")
        
        if chat.debug_mode=="Enable":
            st.info('초안을 검토하여 부족하거나 보강할 내용을 찾고, 추가 검색어를 추출합니다.')

        reflection = []
        search_queries = []
        for attempt in range(5):
            try:
                chat = chat.get_chat(extended_thinking="Disable")
                if chat.isKorean(state["messages"][-1].content):
                    structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
                else:
                    structured_llm = chat.with_structured_output(Research, include_raw=True)
                
                info = structured_llm.invoke(state["messages"][-1].content)
                logger.info(f"attempt: {attempt}, info: {info}")
                    
                if not info['parsed'] == None:
                    parsed_info = info['parsed']
                    # print('reflection: ', parsed_info.reflection)                
                    reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                    search_queries = parsed_info.search_queries
                    
                    logger.info(f"reflection: {parsed_info.reflection}")
                    logger.info(f"search_queries: {search_queries}")

                    if chat.debug_mode=="Enable":  
                        st.info(f'개선할 사항: {parsed_info.reflection}')
                        st.info(f'추가 검색어: {search_queries}')        
                    break
            except Exception:
                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}") 
        
        return {
            "messages": state["messages"],
            "reflection": reflection,
            "search_queries": search_queries
        }
    
    def get_revise_prompt(text):
        llm = chat.get_chat(extended_thinking="Disable")

        if chat.isKorean(text):
            system = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                "critique과 information 참조하여 답변을 수정하십시오."
                "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
            )
            human = (
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )
        else:
            system = (
                "You are an excellent writing assistant." 
                "Revise this draft using the critique and additional information."
                "Provide the final answer with <result> tag."
            )
            human = (
                "critique:"
                "{reflection}"

                "information:"
                "{content}"
            )                    
        revise_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system),
                MessagesPlaceholder(variable_name="messages"),
                ("human", human),
            ]
        )                    
        revise_chain = revise_prompt | llm

        return revise_chain
    
    def revise_answer(state: State, config):   
        logger.info(f"##### revise_answer ######")
        
        if chat.debug_mode=="Enable":
            st.info("개선할 사항을 반영하여 답변을 생성중입니다.")
                    
        content = []        
        if useEnhancedSearch: # search agent
            for q in state["search_queries"]:
                response = search.enhanced_search(q, st)       
                logger.info('q: {q}, response: {response}')

                content.append(response)                   
        else:
            content = search.retrieve_contents_from_tavily(state["search_queries"], top_k=2)

        for attempt in range(5):
            logger.info(f"attempt: {attempt}")
            messages = state["messages"]
            cls_map = {"ai": HumanMessage, "human": AIMessage}
            translated = [messages[0]] + [
                cls_map[msg.type](content=msg.content) for msg in messages[1:]
            ]
            logger.info(f"translated: {translated}")     
            
            revise_chain = get_revise_prompt(content)
            try:
                res = revise_chain.invoke(
                    {
                        "messages": translated,
                        "reflection": state["reflection"],
                        "content": content
                    }
                )    
                output = res.content

                if output.find('<result>')==-1:
                    answer = output
                else:
                    answer = output[output.find('<result>')+8:output.find('</result>')]
            
                response = HumanMessage(content=answer)
                logger.info(f"revised_answer: {response.content}")         
                break

            except Exception:
                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}")            
                
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "messages": [response], 
            "revision_number": revision_number + 1
        }
    
    MAX_REVISIONS = 1
    def should_continue(state: State, config):
        logger.info(f"###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        logger.info(f"prmax_revisionsepare: {max_revisions}")
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"

    def buildKnowledgeGuru():    
        workflow = StateGraph(State)

        workflow.add_node("generate", generate)
        workflow.add_node("reflect", reflect)
        workflow.add_node("revise_answer", revise_answer)

        workflow.set_entry_point("generate")

        workflow.add_conditional_edges(
            "revise_answer", 
            should_continue, 
            {
                "end": END, 
                "continue": "reflect"}
        )

        workflow.add_edge("generate", "reflect")
        workflow.add_edge("reflect", "revise_answer")
        
        app = workflow.compile()
        
        return app
    
    # initiate
    global contentList, reference_docs
    contentList = []
    reference_docs = []

    # workflow
    app = buildKnowledgeGuru()
        
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50,
        "max_revisions": MAX_REVISIONS,
        "parallel_processing": chat.multi_region
    }
    
    for output in app.stream({"messages": inputs}, config):   
        for key, value in output.items():
            logger.info(f"Finished: {key}")
            #print("value: ", value)
            
    logger.info(f"value: {value}")
        
    msg = value["messages"][-1].content

    reference = ""
    if reference_docs:
        reference = chat.get_references(reference_docs)

    return msg+reference, reference_docs

