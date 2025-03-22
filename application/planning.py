import rag_opensearch as rag
import utils
import operator
import chat
import traceback
import search

from typing_extensions import Annotated, TypedDict
from typing import List, Tuple,Literal
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

logger = utils.CreateLogger("planning")

####################### LangGraph #######################
# Agentic Workflow: Planning (Advanced CoT)
#########################################################
def run_planning(query, st):
    class State(TypedDict):
        input: str
        plan: list[str]
        past_steps: Annotated[List[Tuple], operator.add]
        info: Annotated[List[Tuple], operator.add]
        response: list[str]
        answer: str

    def plan_node(state: State, config):
        logger.info(f"###### plan ######")
        logger.info(f"input: {state['input']}")

        if chat.debug_mode=="Enable":
            st.info(f"계획을 생성합니다. 요청사항: {state['input']}")
        
        system = (
            "당신은 user의 question을 해결하기 위해 step by step plan을 생성하는 AI agent입니다."                
            
            "문제를 충분히 이해하고, 문제 해결을 위한 계획을 다음 형식으로 4단계 이하의 계획을 세웁니다."                
            "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
            "1. [질문을 해결하기 위한 단계]"
            "2. [질문을 해결하기 위한 단계]"
            "..."                
        )
        
        human = (
            "{question}"
        )
                            
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )
        llm = chat.get_chat(extended_thinking="Disable")
        planner = planner_prompt | llm
        response = planner.invoke({
            "question": state["input"]
        })
        logger.info(f"response: {response.content}")
        result = response.content
        
        #output = result[result.find('<result>')+8:result.find('</result>')]
        output = result
        
        plan = output.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')
        logger.info(f"planning_steps: {planning_steps}")

        if chat.debug_mode=="Enable":
            st.info(f"생성된 계획: {planning_steps}")
        
        return {
            "input": state["input"],
            "plan": planning_steps
        }
    
    def generate_answer(relevant_docs, text):    
        relevant_context = ""
        for document in relevant_docs:
            relevant_context = relevant_context + document.page_content + "\n\n"        
        # print('relevant_context: ', relevant_context)

        if chat.debug_mode=="Enable":
            st.info(f"계획을 수행합니다. 현재 계획 {text}")

        # generating
        if chat.isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Provide a concise answer to the question at the end using reference texts." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            )    
        human = (
            "Question: {input}"

            "Reference texts: "
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
                        
        llm = chat.get_chat(extended_thinking="Disable")
        chain = prompt | llm
        
        response = chain.invoke({
            "context": relevant_context,
            "input": text,
        })
        # print('response.content: ', response.content)

        if response.content.find('<result>') == -1:
            output = response.content
        else:
            output = response.content[response.content.find('<result>')+8:response.content.find('</result>')]        
        # print('output: ', output)
            
        return output

    def execute_node(state: State, config):
        logger.info(f"###### execute ######")
        logger.info(f"input: {state['input']}")
        plan = state["plan"]
        logger.info(f"plan: {plan}")
        
        if chat.debug_mode=="Enable":
            st.info(f"검색을 수행합니다. 검색어 {plan[0]}")
        
        # retrieve
        relevant_docs = rag.retrieve_documents_from_opensearch(plan[0], top_k=4)   
        if chat.internet_mode == "Enable":  
            relevant_docs += search.retrieve_documents_from_tavily(plan[0], top_k=4)
        
        # grade   
        if chat.debug_mode == "Enable":
            st.info(f"가져온 {len(relevant_docs)}개의 문서를 평가하고 있습니다.") 

        filtered_docs = chat.grade_documents(plan[0], relevant_docs) # grading    
        filtered_docs = chat.check_duplication(filtered_docs) # check duplication

        global reference_docs
        if len(filtered_docs):
            reference_docs += filtered_docs

        if chat.debug_mode == "Enable":
            st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")
                
        # generate
        if chat.debug_mode == "Enable":
            st.info(f"결과를 생성중입니다.")

        result = generate_answer(relevant_docs, plan[0])
        
        logger.info(f"task: {plan[0]}")
        logger.info(f"executor output: {result}")

        if chat.debug_mode=="Enable":
            st.info(f"현 단계의 결과 {result}")
        
        # print('plan: ', state["plan"])
        # print('past_steps: ', task)        
        return {
            "input": state["input"],
            "plan": state["plan"],
            "info": [result],
            "past_steps": [plan[0]],
        }
            
    def replan_node(state: State, config):
        logger.info(f"#### replan ####")
        logger.info(f"state of replan node: {state}")

        if len(state["plan"]) == 1:
            logger.info(f"last plan: {state['plan']}")
            logger.info(f"final info: {state['info'][-1]}")
            return {"response":state['info'][-1]}
        
        if chat.debug_mode=="Enable":
            st.info(f"새로운 계획을 생성합니다.")
        
        system = (
            "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
            "당신은 다음의 Question에 대한 적절한 답변을 얻고자합니다."
        )        
        human = (
            "Question: {input}"
                        
            "당신의 원래 계획은 아래와 같습니다." 
            "Original Plan:"
            "{plan}"

            "완료한 계획는 아래와 같습니다."
            "Past steps:"
            "{past_steps}"
            
            "당신은 Original Plan의 원래 계획을 상황에 맞게 수정하세요."
            "계획에 아직 해야 할 단계만 추가하세요. 이전에 완료한 계획는 계획에 포함하지 마세요."                
            "수정된 계획에는 <plan> tag를 붙여주세요."
            "만약 더 이상 계획을 세우지 않아도 Question에 답변할 수 있다면, 최종 결과로 Question에 대한 답변을 <result> tag를 붙여 전달합니다."
            
            "수정된 계획의 형식은 아래와 같습니다."
            "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
            "1. [질문을 해결하기 위한 단계]"
            "2. [질문을 해결하기 위한 단계]"
            "..."         
        )                   
        
        replanner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )     
        
        llm = chat.get_chat(extended_thinking="Disable")
        replanner = replanner_prompt | llm
        
        response = replanner.invoke({
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": state["past_steps"]
        })
        logger.info(f"replanner output:: {response.content}")
        result = response.content

        if result.find('<plan>') == -1:
            return {"response":response.content}
        else:
            output = result[result.find('<plan>')+6:result.find('</plan>')]
            logger.info(f"plan output: {output}")

            plans = output.strip().replace('\n\n', '\n')
            planning_steps = plans.split('\n')
            logger.info(f"planning_steps: {planning_steps}")

            if chat.debug_mode=="Enable":
                st.info(f"새로운 계획: {planning_steps}")

            return {"plan": planning_steps}
        
    def should_end(state: State) -> Literal["continue", "end"]:
        logger.info(f"#### should_end ####")
        logger.info(f"state: {state}")
        
        if "response" in state and state["response"]:
            logger.info(f"response: {state['response']}")
            next = "end"
        else:
            logger.info(f"plan: {state['plan']}")
            next = "continue"
        logger.info(f"should_end response: {next}")
        
        return next
        
    def final_answer(state: State) -> str:
        logger.info(f"#### final_answer ####")
        
        # get final answer
        context = "".join(f"{info}\n" for info in state['info'])
        logger.info(f"context: {context}")
        
        query = state['input']
        logger.info(f"query: {query}")

        if chat.debug_mode=="Enable":
            st.info(f"최종 답변을 생성합니다.")
        
        if chat.isKorean(query)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                #"결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "Here is pieces of context, contained in <context> tags."
                "Provide a concise answer to the question at the end."
                "Explains clearly the reason for the answer."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                #"Put it in <result> tags."
            )
    
        human = (
            "Reference texts:"
            "{context}"

            "Question: {input}"
        )
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
                    
        llm = chat.get_chat(extended_thinking="Disable")
        chain = prompt | llm
        
        try: 
            response = chain.invoke(
                {
                    "context": context,
                    "input": query,
                }
            )
            result = response.content

            # extended thinking
            if chat.debug_mode=="Enable":
                chat.show_extended_thinking(st, response)

            if result.find('<result>')==-1:
                output = result
            else:
                output = result[result.find('<result>')+8:result.find('</result>')]
                
            logger.info(f"output: {output}")
            
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")      
            
        return {"answer": output}  

    def buildPlanAndExecute():
        workflow = StateGraph(State)
        workflow.add_node("planner", plan_node)
        workflow.add_node("executor", execute_node)
        workflow.add_node("replaner", replan_node)
        workflow.add_node("final_answer", final_answer)
        
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "replaner")
        workflow.add_conditional_edges(
            "replaner",
            should_end,
            {
                "continue": "executor",
                "end": "final_answer",
            },
        )
        workflow.add_edge("final_answer", END)

        return workflow.compile()

    # initiate
    global contentList, reference_docs
    contentList = []
    reference_docs = []

    # workflow
    app = buildPlanAndExecute()    
        
    inputs = {"input": query}
    config = {
        "recursion_limit": 50
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            logger.info(f"Finished: {key}")
            #print("value: ", value)            
    logger.info(f"value: {value}")

    reference = ""
    if reference_docs:
        reference = chat.get_references(reference_docs)
    
    return value["answer"]+reference, reference_docs
