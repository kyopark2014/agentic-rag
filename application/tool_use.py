import traceback
import json
import re
import requests
import datetime
import yfinance as yf
import utils
import chat
import rag_opensearch as rag
import search

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from bs4 import BeautifulSoup
from pytz import timezone
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langchain.docstore.document import Document

logger = utils.CreateLogger("tool-use")

####################### LangGraph #######################
# Agentic RAG
#########################################################
@tool 
def get_book_list(keyword: str) -> str:
    """
    Search book list by keyword and then return book list
    keyword: search keyword
    return: book list
    """
    
    keyword = keyword.replace('\'','')

    answer = ""
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        
        if len(prod_info):
            answer = "추천 도서는 아래와 같습니다.\n"
            
        for prod in prod_info[:5]:
            title = prod.text.strip().replace("\n", "")       
            link = prod.get("href")
            answer = answer + f"{title}, URL: {link}\n\n"
    
    return answer

@tool
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    # f"%Y-%m-%d %H:%M:%S"
    
    format = format.replace('\'','')
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    logger.info(f"timestr: {timestr}")
    
    return timestr

@tool
def get_weather_info(city: str) -> str:
    """
    retrieve weather information by city name and then return weather statement.
    city: the name of city to retrieve
    return: weather statement
    """    
    
    city = city.replace('\n','')
    city = city.replace('\'','')
    city = city.replace('\"','')
                
    llm = chat.get_chat()
    if chat.isKorean(city):
        place = chat.traslation(llm, city, "Korean", "English")
        logger.info(f"city (translated): ", place)
    else:
        place = city
        city = chat.traslation(llm, city, "English", "Korean")
        logger.info(f"city (translated): {city}")
        
    logger.info(f"place: {place}")
    
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."
    if chat.weather_api_key: 
        apiKey = chat.weather_api_key
        lang = 'en' 
        units = 'metric' 
        api = f"https://api.openweathermap.org/data/2.5/weather?q={place}&APPID={apiKey}&lang={lang}&units={units}"
        # print('api: ', api)
                
        try:
            result = requests.get(api)
            result = json.loads(result.text)
            logger.info(f"result: {result}")
        
            if 'weather' in result:
                overall = result['weather'][0]['main']
                current_temp = result['main']['temp']
                min_temp = result['main']['temp_min']
                max_temp = result['main']['temp_max']
                humidity = result['main']['humidity']
                wind_speed = result['wind']['speed']
                cloud = result['clouds']['all']
                
                #weather_str = f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp}도 이고, 최저온도는 {min_temp}도, 최고 온도는 {max_temp}도 입니다. 현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. 구름은 {cloud}% 입니다."
                weather_str = f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp} 입니다. 현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. 구름은 {cloud}% 입니다."
                #weather_str = f"Today, the overall of {city} is {overall}, current temperature is {current_temp} degree, min temperature is {min_temp} degree, highest temperature is {max_temp} degree. huminity is {humidity}%, wind status is {wind_speed} meter per second. the amount of cloud is {cloud}%."            
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")   
            # raise Exception ("Not able to request to LLM")    
        
    logger.info(f"weather_str: {weather_str}")                        
    return weather_str

@tool    
def search_by_opensearch(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    
    
    logger.info(f"keyword: {keyword}")
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    logger.info(f"modified keyword: {keyword}")
    
    # retrieve
    relevant_docs = rag.retrieve_documents_from_opensearch(keyword, top_k=2)                            
    logger.info(f"relevant_docs length: {len(relevant_docs)}")

    # grade  
    filtered_docs = chat.grade_documents(keyword, relevant_docs)

    global reference_docs
    if len(filtered_docs):
        reference_docs += filtered_docs
        
    for i, doc in enumerate(filtered_docs):
        if len(doc.page_content)>=100:
            text = doc.page_content[:100]
        else:
            text = doc.page_content            
        logger.info(f"filtered doc[{i}]: {text}, metadata:{doc.metadata}")
       
    relevant_context = "" 
    for doc in filtered_docs:
        content = doc.page_content
        
        relevant_context = relevant_context + f"{content}\n\n"

    if len(filtered_docs) == 0:
        #relevant_context = "No relevant documents found."
        relevant_context = "관련된 정보를 찾지 못하였습니다."
        logger.info(f"relevant_context: {relevant_context}")
        
    return relevant_context
    
@tool
def search_by_tavily(keyword: str) -> str:
    """
    Search general knowledge by keyword and then return the result as a string.
    keyword: search keyword
    return: the information of keyword
    """    
    global reference_docs    
    answer = ""
    
    keyword = keyword.replace('\'','')
    relevant_documents = search.retrieve_documents_from_tavily(keyword, top_k=3)

    answer = ""
    for doc in reference_docs:
        content = doc.page_content
        url = doc.metadata['url']
        answer += + f"{content}, URL: {url}\n" 

    if len(relevant_documents):
        reference_docs += relevant_documents

    if answer == "":
        # answer = "No relevant documents found." 
        answer = "관련된 정보를 찾지 못하였습니다."
                     
    return answer

@tool
def stock_data_lookup(ticker, country):
    """
    Retrieve accurate stock trends for a given ticker.
    ticker: the ticker to retrieve price history for
    country: the english country name of the stock
    return: the information of ticker
    """ 
    com = re.compile('[a-zA-Z]') 
    alphabet = com.findall(ticker)
    logger.info(f"alphabet: {alphabet}")

    logger.info(f"country: {country}")

    if len(alphabet)==0:
        if country == "South Korea":
            ticker += ".KS"
        elif country == "Japan":
            ticker += ".T"
    logger.info(f"ticker: {ticker}")
    
    stock = yf.Ticker(ticker)
    
    # get the price history for past 1 month
    history = stock.history(period="1mo")
    logger.info(f"history: {history}")
    
    result = f"## Trading History\n{history}"
    #history.reset_index().to_json(orient="split", index=False, date_format="iso")    
    
    result += f"\n\n## Financials\n{stock.financials}"    
    logger.info(f"financials: {stock.financials}")

    result += f"\n\n## Major Holders\n{stock.major_holders}"
    logger.info(f"major_holders: {stock.major_holders}")

    logger.info(f"result: {result}")

    return result

tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch, stock_data_lookup]        

def run_agent_executor(query, st):
    chatModel = chat.get_chat()     
    model = chatModel.bind_tools(tools)

    class State(TypedDict):
        # messages: Annotated[Sequence[BaseMessage], operator.add]
        messages: Annotated[list, add_messages]

    tool_node = ToolNode(tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        logger.info(f"###### should_continue ######")

        logger.info(f"state: {state}")
        messages = state["messages"]    

        last_message = messages[-1]
        logger.info(f"last_message: {last_message}")
        
        # print('last_message: ', last_message)
        
        # if not isinstance(last_message, ToolMessage):
        #     return "end"
        # else:                
        #     return "continue"
        if isinstance(last_message, ToolMessage) or last_message.tool_calls:
            logger.info(f"tool_calls: {last_message.tool_calls}")

            for message in last_message.tool_calls:
                logger.info(f"tool name: {message['name']}, args: {message['args']}")
                # update_state_message(f"calling... {message['name']}", config)

            logger.info(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"
        
        #if not last_message.tool_calls:
        else:
            logger.info(f"Final: {last_message.content}")
            logger.info(f"--- END ---")
            return "end"
           
    def call_model(state: State, config):
        print("###### call_model ######")
        logger.info(f"###### call_model ######")
        logger.info(f"state: {state['messages']}")
                
        if chat.isKorean(state["messages"][0].content)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
                "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "한국어로 답변하세요."
            )
        else: 
            system = (            
                "You are a conversational AI designed to answer in a friendly way to a question."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            )

        for attempt in range(3):   
            logger.info(f"attempt: {attempt}")
            try:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                chain = prompt | model
                    
                response = chain.invoke(state["messages"])
                logger.info(f"all_model response: {response}")

                if isinstance(response.content, list):            
                    for re in response.content:
                        if "type" in re:
                            if re['type'] == 'text':
                                logger.info(f"--> {re['type']}: {re['text']}")

                                status = re['text']
                                logger.info(f"status: {status}")
                                
                                status = status.replace('`','')
                                status = status.replace('\"','')
                                status = status.replace("\'",'')
                                
                                logger.info(f"status: {status}")
                                if status.find('<thinking>') != -1:
                                    logger.info(f"Remove <thinking> tag.")
                                    status = status[status.find('<thinking>')+11:status.find('</thinking>')]
                                    logger.info(f"tatus without tag: {status}")

                                if chat.debug_mode=="Enable":
                                    st.info(status)
                                
                            elif re['type'] == 'tool_use':                
                                logger.info(f"--> {re['type']}: {re['name']}, {re['input']}")

                                if chat.debug_mode=="Enable":
                                    st.info(f"{re['type']}: {re['name']}, {re['input']}")
                            else:
                                print(re)
                                logger.info(f"{re}")
                        else: # answer
                            print(response.content)
                            logger.info(f"{response.content}")
                break
            except Exception:
                response = AIMessage(content="답변을 찾지 못하였습니다.")

                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}")
                # raise Exception ("Not able to request to LLM")

        return {"messages": [response]}

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
        workflow.add_edge(START, "agent")
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

    # initiate
    global reference_docs, contentList
    reference_docs = []
    contentList = []

    # workflow 
    app = buildChatAgent()
            
    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50
    }
    
    # msg = message.content
    result = app.invoke({"messages": inputs}, config)
    #print("result: ", result)

    msg = result["messages"][-1].content
    logger.info(f"msg: {msg}")

    for i, doc in enumerate(reference_docs):
        logger.info(f"--> reference {i}: {doc}")
        
    reference = ""
    if reference_docs:
        reference = chat.get_references(reference_docs)

    msg = chat.extract_thinking_tag(msg, st)
    
    return msg+reference, reference_docs
