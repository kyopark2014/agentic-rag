import traceback
import boto3
import os
import json
import re
import requests
import datetime
import functools
import uuid
import time
import logging
import base64
import operator

from io import BytesIO
from PIL import Image
from pytz import timezone
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool
from langchain.docstore.document import Document
from tavily import TavilyClient  
from langchain_community.tools.tavily_search import TavilySearchResults
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from typing import Any, List, Tuple, Dict, Optional, cast, Literal, Sequence, Union
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_aws import AmazonKnowledgeBasesRetriever
from multiprocessing import Process, Pipe
from urllib import parse
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

try:
    with open("/home/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        print('config: ', config)
except Exception:
    print("use local configuration")
    with open("application/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        print('config: ', config)

bedrock_region = config["region"] if "region" in config else "us-west-2"

projectName = config["projectName"] if "projectName" in config else "langgraph-nova"

accountId = config["accountId"] if "accountId" in config else None
if accountId is None:
    raise Exception ("No accountId")

region = config["region"] if "region" in config else "us-west-2"
print('region: ', region)

bucketName = config["bucketName"] if "bucketName" in config else f"storage-for-{projectName}-{accountId}-{region}" 
print('bucketName: ', bucketName)

s3_prefix = 'docs'

knowledge_base_role = config["knowledge_base_role"] if "knowledge_base_role" in config else None
if knowledge_base_role is None:
    raise Exception ("No Knowledge Base Role")

collectionArn = config["collectionArn"] if "collectionArn" in config else None
if collectionArn is None:
    raise Exception ("No collectionArn")

vectorIndexName = projectName

opensearch_url = config["opensearch_url"] if "opensearch_url" in config else None
if opensearch_url is None:
    raise Exception ("No OpenSearch URL")

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

credentials = boto3.Session().get_credentials()
service = "aoss" 
awsauth = AWSV4SignerAuth(credentials, region, service)

s3_arn = config["s3_arn"] if "s3_arn" in config else None
if s3_arn is None:
    raise Exception ("No S3 ARN")

parsingModelArn = f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
embeddingModelArn = f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"

knowledge_base_name = projectName

numberOfDocs = 4
MSG_LENGTH = 100    
grade_state = "LLM" # LLM, PRIORITY_SEARCH, OTHERS
parallel_processing = 'enable'

doc_prefix = s3_prefix+'/'
useEnhancedSearch = False

parallel_processing_models = [   # Nova Pro
    {   
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    }
]
selected_chat = 0
number_of_models = len(parallel_processing_models)
STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'

userId = "demo"
map_chain = dict() 

def initiate():
    global userId
    global memory_chain

    userId = uuid.uuid4().hex
    print('userId: ', userId)

    if userId in map_chain:  
            # print('memory exist. reuse it!')
            memory_chain = map_chain[userId]
    else: 
        # print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

initiate()

def clear_chat_history():
    memory_chain = []
    map_chain[userId] = memory_chain

def save_chat_history(text, msg):
    memory_chain.chat_memory.add_user_message(text)
    if len(msg) > MSG_LENGTH:
        memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
    else:
        memory_chain.chat_memory.add_ai_message(msg) 

def get_chat():
    global selected_chat
    
    profile = parallel_processing_models[selected_chat]
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [STOP_SEQUENCE]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    )    
    
    selected_chat = selected_chat + 1
    if selected_chat == number_of_models:
        selected_chat = 0
    
    return chat

def get_parallel_processing_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [STOP_SEQUENCE]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )        
    return chat

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")

reference_docs = []
# api key to get weather information in agent
secretsmanager = boto3.client(
    service_name='secretsmanager',
    region_name=bedrock_region
)
try:
    get_weather_api_secret = secretsmanager.get_secret_value(
        SecretId=f"openweathermap-{projectName}"
    )
    #print('get_weather_api_secret: ', get_weather_api_secret)
    if get_weather_api_secret['SecretString']:
        secret = json.loads(get_weather_api_secret['SecretString'])
        #print('secret: ', secret)
        weather_api_key = secret['weather_api_key']
    else:
        print('No secret found for weather api')

except Exception as e:
    raise e

# api key to use LangSmith
langsmith_api_key = ""
try:
    get_langsmith_api_secret = secretsmanager.get_secret_value(
        SecretId=f"langsmithapikey-{projectName}"
    )
    #print('get_langsmith_api_secret: ', get_langsmith_api_secret)
    if get_langsmith_api_secret['SecretString']:
        secret = json.loads(get_langsmith_api_secret['SecretString'])
        #print('secret: ', secret)
        langsmith_api_key = secret['langsmith_api_key']
        langchain_project = secret['langchain_project']
    else:
        print('No secret found for lengsmith api')
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

# api key to use Tavily Search
tavily_key = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    #print('secret: ', secret)
    tavily_key = secret['tavily_api_key']
    #print('tavily_api_key: ', tavily_api_key)

    tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
    #     os.environ["TAVILY_API_KEY"] = tavily_key

    # Tavily Tool Test
    query = 'what is Amazon Nova Pro?'
    search = TavilySearchResults(
        max_results=1,
        include_answer=True,
        include_raw_content=True,
        api_wrapper=tavily_api_wrapper,
        search_depth="advanced", # "basic"
        # include_domains=["google.com", "naver.com"]
    )
    output = search.invoke(query)
    print('tavily output: ', output)    
except Exception as e: 
    print('Tavily credential is required: ', e)
    raise e

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False

def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags." 
        "Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)          
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag
    
def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_parallel_processing_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()

def grade_documents_using_parallel_processing(question, documents):
    global selected_chat
    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, parallel_processing_models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()

        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()
    
    return filtered_docs

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader(chat):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    print("start grading...")
    print("grade_state: ", grade_state)
    
    if grade_state == "LLM":
        filtered_docs = []
        if parallel_processing == 'enable':  # parallel processing        
            filtered_docs = grade_documents_using_parallel_processing(question, documents)

        else:
            # Score each doc    
            chat = get_chat()
            retrieval_grader = get_retrieval_grader(chat)
            for i, doc in enumerate(documents):
                # print('doc: ', doc)
                print_doc(i, doc)
                
                score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                # print("score: ", score)
                
                grade = score.binary_score
                # print("grade: ", grade)
                # Document relevant
                if grade.lower() == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(doc)
                # Document not relevant
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    # We do not include the document in filtered_docs
                    # We set a flag to indicate that we want to run web search
                    continue
    
    else:  # OTHERS
        filtered_docs = documents

    return filtered_docs

contentList = []
def check_duplication(docs):
    global contentList
    length_original = len(docs)
    
    updated_docs = []
    print('length of relevant_docs:', len(docs))
    for doc in docs:            
        if doc.page_content in contentList:
            print('duplicated!')
            continue
        contentList.append(doc.page_content)
        updated_docs.append(doc)            
    length_updateed_docs = len(updated_docs)     
    
    if length_original == length_updateed_docs:
        print('no duplication')
    
    return updated_docs

def retrieve_documents_from_tavily(query, top_k):
    print("###### retrieve_documents_from_tavily ######")

    relevant_documents = []        
    search = TavilySearchResults(
        max_results=top_k,
        include_answer=True,
        include_raw_content=True,        
        api_wrapper=tavily_api_wrapper,
        search_depth="advanced", 
        # include_domains=["google.com", "naver.com"]
    )
                    
    try: 
        output = search.invoke(query)
        print('tavily output: ', output)

        #if output.find("429 Client Error: Too Many Requests for url"):
        #    raise Exception ("Tavily error: Too Many Requests for url")
            
        print(f"--> tavily query: {query}")
        for i, result in enumerate(output):
            print(f"{i}: {result}")
            if result:
                content = result.get("content")
                url = result.get("url")
                
                relevant_documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            'name': 'WWW',
                            'url': url,
                            'from': 'tavily'
                        },
                    )
                )                
    
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        # raise Exception ("Not able to request to tavily")   

    return relevant_documents 

def get_references(docs):    
    reference = ""
    for i, doc in enumerate(docs):
        page = ""
        if "page" in doc.metadata:
            page = doc.metadata['page']
            #print('page: ', page)            
        url = ""
        if "url" in doc.metadata:
            url = doc.metadata['url']
            print('url: ', url)
        name = ""
        if "name" in doc.metadata:
            name = doc.metadata['name']
            #print('name: ', name)     
        
        sourceType = ""
        if "from" in doc.metadata:
            sourceType = doc.metadata['from']
        else:
            # if useEnhancedSearch:
            #     sourceType = "OpenSearch"
            # else:
            #     sourceType = "WWW"
            sourceType = "WWW"

        #print('sourceType: ', sourceType)        
        
        #if len(doc.page_content)>=1000:
        #    excerpt = ""+doc.page_content[:1000]
        #else:
        #    excerpt = ""+doc.page_content
        excerpt = ""+doc.page_content
        # print('excerpt: ', excerpt)
        
        # for some of unusual case 
        #excerpt = excerpt.replace('"', '')        
        #excerpt = ''.join(c for c in excerpt if c not in '"')
        excerpt = re.sub('"', '', excerpt)
        excerpt = re.sub('#', '', excerpt)        
        print('excerpt(quotation removed): ', excerpt)
        
        if page:                
            reference += f"{i+1}. {page}page in [{name}]({url})), {excerpt[:40]}...\n"
        else:
            reference += f"{i+1}. [{name}]({url}), {excerpt[:40]}...\n"

    if reference: 
        reference = "\n\n#### 관련 문서\n"+reference

    return reference

def tavily_search(query, k):
    docs = []    
    try:
        tavily_client = TavilyClient(
            api_key=tavily_key
        )
        response = tavily_client.search(query, max_results=k)
        # print('tavily response: ', response)
            
        for r in response["results"]:
            name = r.get("title")
            if name is None:
                name = 'WWW'
            
            docs.append(
                Document(
                    page_content=r.get("content"),
                    metadata={
                        'name': name,
                        'url': r.get("url"),
                        'from': 'tavily'
                    },
                )
            )                   
    except Exception as e:
        print('Exception: ', e)

    return docs

####################### LangChain #######################
# General Conversation
#########################################################

def general_conversation(query):
    chat = get_chat()

    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
    
    human = "Question: {input}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)
    ])
                
    history = memory_chain.load_memory_variables({})["chat_history"]

    chain = prompt | chat | StrOutputParser()
    try: 
        stream = chain.stream(
            {
                "history": history,
                "input": query,
            }
        )  
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to request to LLM: "+err_msg)
        
    return stream

    
####################### LangGraph #######################
# RAG: Knowledge Base
#########################################################

os_client = OpenSearch(
    hosts = [{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_auth=awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class=RequestsHttpConnection,
)

def is_not_exist(index_name):    
    print('index_name: ', index_name)
        
    if os_client.indices.exists(index_name):
        print('use exist index: ', index_name)    
        return False
    else:
        print('no index: ', index_name)
        return True
    
knowledge_base_id = ""
data_source_id = ""
def initiate_knowledge_base():
    global knowledge_base_id, data_source_id
    #########################
    # opensearch index
    #########################
    if(is_not_exist(vectorIndexName)):
        print(f"creating opensearch index... {vectorIndexName}")        
        body={ 
            'settings':{
                "index.knn": True,
                "index.knn.algo_param.ef_search": 512,
                'analysis': {
                    'analyzer': {
                        'my_analyzer': {
                            'char_filter': ['html_strip'], 
                            'tokenizer': 'nori',
                            'filter': ['nori_number','lowercase','trim','my_nori_part_of_speech'],
                            'type': 'custom'
                        }
                    },
                    'tokenizer': {
                        'nori': {
                            'decompound_mode': 'mixed',
                            'discard_punctuation': 'true',
                            'type': 'nori_tokenizer'
                        }
                    },
                    "filter": {
                        "my_nori_part_of_speech": {
                            "type": "nori_part_of_speech",
                            "stoptags": [
                                    "E", "IC", "J", "MAG", "MAJ",
                                    "MM", "SP", "SSC", "SSO", "SC",
                                    "SE", "XPN", "XSA", "XSN", "XSV",
                                    "UNA", "NA", "VSV"
                            ]
                        }
                    }
                },
            },
            'mappings': {
                'properties': {
                    'vector_field': {
                        'type': 'knn_vector',
                        'dimension': 1024,
                        'method': {
                            "name": "hnsw",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            }
                        }                  
                    },
                    "AMAZON_BEDROCK_METADATA": {"type": "text", "index": False},
                    "AMAZON_BEDROCK_TEXT": {"type": "text"},
                }
            }
        }

        try: # create index
            response = os_client.indices.create(
                vectorIndexName,
                body=body
            )
            print('opensearch index was created:', response)

            # delay 3seconds
            time.sleep(5)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                
            #raise Exception ("Not able to create the index")
            
    #########################
    # knowledge base
    #########################
    print('knowledge_base_name: ', knowledge_base_name)
    print('collectionArn: ', collectionArn)
    print('vectorIndexName: ', vectorIndexName)
    print('embeddingModelArn: ', embeddingModelArn)
    print('knowledge_base_role: ', knowledge_base_role)
    try: 
        client = boto3.client(
            service_name='bedrock-agent',
            region_name=bedrock_region
        )   

        response = client.list_knowledge_bases(
            maxResults=10
        )
        print('(list_knowledge_bases) response: ', response)
        
        if "knowledgeBaseSummaries" in response:
            summaries = response["knowledgeBaseSummaries"]
            for summary in summaries:
                if summary["name"] == knowledge_base_name:
                    knowledge_base_id = summary["knowledgeBaseId"]
                    print('knowledge_base_id: ', knowledge_base_id)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)
                    
    if not knowledge_base_id:
        print('creating knowledge base...')        
        for atempt in range(20):
            try:
                response = client.create_knowledge_base(
                    name=knowledge_base_name,
                    description="Knowledge base based on OpenSearch",
                    roleArn=knowledge_base_role,
                    knowledgeBaseConfiguration={
                        'type': 'VECTOR',
                        'vectorKnowledgeBaseConfiguration': {
                            'embeddingModelArn': embeddingModelArn,
                            'embeddingModelConfiguration': {
                                'bedrockEmbeddingModelConfiguration': {
                                    'dimensions': 1024
                                }
                            }
                        }
                    },
                    storageConfiguration={
                        'type': 'OPENSEARCH_SERVERLESS',
                        'opensearchServerlessConfiguration': {
                            'collectionArn': collectionArn,
                            'fieldMapping': {
                                'metadataField': 'AMAZON_BEDROCK_METADATA',
                                'textField': 'AMAZON_BEDROCK_TEXT',
                                'vectorField': 'vector_field'
                            },
                            'vectorIndexName': vectorIndexName
                        }
                    }                
                )   
                print('(create_knowledge_base) response: ', response)
            
                if 'knowledgeBaseId' in response['knowledgeBase']:
                    knowledge_base_id = response['knowledgeBase']['knowledgeBaseId']
                    break
                else:
                    knowledge_base_id = ""    
            except Exception:
                    err_msg = traceback.format_exc()
                    print('error message: ', err_msg)
                    time.sleep(5)
                    print(f"retrying... ({atempt})")
                    #raise Exception ("Not able to create the knowledge base")      
                
    print(f"knowledge_base_name: {knowledge_base_name}, knowledge_base_id: {knowledge_base_id}")    
    
    #########################
    # data source      
    #########################
    data_source_name = bucketName  
    try: 
        response = client.list_data_sources(
            knowledgeBaseId=knowledge_base_id,
            maxResults=10
        )        
        print('(list_data_sources) response: ', response)
        
        if 'dataSourceSummaries' in response:
            for data_source in response['dataSourceSummaries']:
                print('data_source: ', data_source)
                if data_source['name'] == data_source_name:
                    data_source_id = data_source['dataSourceId']
                    print('data_source_id: ', data_source_id)
                    break    
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)
        
    if not data_source_id:
        print('creating data source...')  
        try:
            response = client.create_data_source(
                dataDeletionPolicy='DELETE',
                dataSourceConfiguration={
                    's3Configuration': {
                        'bucketArn': s3_arn,
                        'inclusionPrefixes': [ 
                            s3_prefix+'/',
                        ]
                    },
                    'type': 'S3'
                },
                description = f"S3 data source: {bucketName}",
                knowledgeBaseId = knowledge_base_id,
                name = data_source_name,
                vectorIngestionConfiguration={
                    'chunkingConfiguration': {
                        'chunkingStrategy': 'HIERARCHICAL',
                        'hierarchicalChunkingConfiguration': {
                            'levelConfigurations': [
                                {
                                    'maxTokens': 1500
                                },
                                {
                                    'maxTokens': 300
                                }
                            ],
                            'overlapTokens': 60
                        }
                    },
                    'parsingConfiguration': {
                        'bedrockFoundationModelConfiguration': {
                            'modelArn': parsingModelArn
                        },
                        'parsingStrategy': 'BEDROCK_FOUNDATION_MODEL'
                    }
                }
            )
            print('(create_data_source) response: ', response)
            
            if 'dataSource' in response:
                if 'dataSourceId' in response['dataSource']:
                    data_source_id = response['dataSource']['dataSourceId']
                    print('data_source_id: ', data_source_id)
                    
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            #raise Exception ("Not able to create the data source")
    
    print(f"data_source_name: {data_source_name}, data_source_id: {data_source_id}")
            
initiate_knowledge_base()

def retrieve_documents_from_knowledge_base(query, top_k):
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID"   # SEMANTIC
            }},
            region_name=bedrock_region
        )
        
        try: 
            documents = retriever.invoke(query)
            # print('documents: ', documents)
            print('--> docs from knowledge base')
            for i, doc in enumerate(documents):
                print_doc(i, doc)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)    
            raise Exception ("Not able to request to LLM: "+err_msg)
        
        relevant_docs = []
        for doc in documents:
            content = ""
            if doc.page_content:
                content = doc.page_content
            
            score = doc.metadata["score"]
            
            link = ""
            if "s3Location" in doc.metadata["location"]:
                link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""
                
                # print('link:', link)    
                pos = link.find(f"/{doc_prefix}")
                name = link[pos+len(doc_prefix)+1:]
                encoded_name = parse.quote(name)
                # print('name:', name)
                link = f"{path}{doc_prefix}{encoded_name}"
                
            elif "webLocation" in doc.metadata["location"]:
                link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
                name = "WEB"

            url = link
            # print('url:', url)
            
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'score': score,
                        'url': url,
                        'from': 'RAG'
                    },
                )
            )    
    return relevant_docs

def generate_answer_using_RAG(chat, context, question):    
    if isKorean(question)==True:
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
            "You will only answer in text format, using markdown format is not allowed."
            "Put it in <result> tags."
        )
    
    human = (
        "Question: {input}"

        "Reference texts: "
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
                   
    chain = prompt | chat
    
    try: 
        output = chain.invoke(
            {
                "context": context,
                "input": question,
            }
        )
        msg = output.content
        print('msg: ', msg)

        if msg.find('<result>')!=-1:
            msg = msg[msg.find('<result>')+8:msg.find('</result>')]
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)    
        raise Exception ("Not able to request to LLM: "+err_msg)

    return msg

def run_rag_with_knowledge_base(text, st, debugMode):
    global reference_docs, contentList
    reference_docs = []
    contentList = []

    chat = get_chat()
    
    msg = ""
    top_k = numberOfDocs
    
    # retrieve
    if debugMode == "Debug":
        st.info(f"검색을 수행합니다. 검색어: {text}")
    
    relevant_docs = retrieve_documents_from_knowledge_base(text, top_k=top_k)
    # relevant_docs += retrieve_documents_from_tavily(text, top_k=top_k)

    # grade   
    if debugMode == "Debug":
        st.info(f"가져온 문서를 평가하고 있습니다.")

    # docs = []
    # for doc in relevant_docs:
    #     chat = get_chat()
    #     if not isKorean(doc.page_content):
    #         translated_content = traslation(chat, doc.page_content, "English", "Korean")
    #         doc.page_content = translated_content
    #         print("doc.page_content: ", doc.page_content)
    #     docs.append(doc)
    # print('translated relevant docs: ', docs)

    filtered_docs = grade_documents(text, relevant_docs)
    
    filtered_docs = check_duplication(filtered_docs) # duplication checker
            
    relevant_context = ""
    for i, document in enumerate(filtered_docs):
        print(f"{i}: {document}")
        if document.page_content:
            content = document.page_content
            
        relevant_context = relevant_context + content + "\n\n"
        
    print('relevant_context: ', relevant_context)

    # generate
    if debugMode == "Debug":
        st.info(f"결과를 생성중입니다.")

    msg = generate_answer_using_RAG(chat, relevant_context, text)
    
    if len(filtered_docs):
        reference_docs += filtered_docs 

    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)

    return msg+reference

def extract_thinking_tag(msg, st, debugMode):
    if msg.find('<thinking>') != -1:
        print('Remove <thinking> tag.')
        status = msg[msg.find('<thinking>')+11:msg.find('</thinking>')]
        print('status without tag: ', status)
        msg = msg[msg.find('</thinking>')+12:]

        if debugMode=="Debug":
            st.info(status)
    return msg

####################### LangGraph #######################
# Agentic Workflow: Tool Use
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
    print('timestr:', timestr)
    
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
                
    chat = get_chat()
    if isKorean(city):
        place = traslation(chat, city, "Korean", "English")
        print('city (translated): ', place)
    else:
        place = city
        city = traslation(chat, city, "English", "Korean")
        print('city (translated): ', city)
        
    print('place: ', place)
    
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."
    if weather_api_key: 
        apiKey = weather_api_key
        lang = 'en' 
        units = 'metric' 
        api = f"https://api.openweathermap.org/data/2.5/weather?q={place}&APPID={apiKey}&lang={lang}&units={units}"
        # print('api: ', api)
                
        try:
            result = requests.get(api)
            result = json.loads(result.text)
            print('result: ', result)
        
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
            print('error message: ', err_msg)   
            # raise Exception ("Not able to request to LLM")    
        
    print('weather_str: ', weather_str)                            
    return weather_str

# Tavily Tool
tavily_tool = TavilySearchResults(
    max_results=3,
    include_answer=True,
    include_raw_content=True,
    api_wrapper=tavily_api_wrapper,
    search_depth="advanced", # "basic"
    # include_domains=["google.com", "naver.com"]
)
     
@tool    
def search_by_knowledge_base(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    
    print("###### search_by_knowledge_base ######")    
    
    global reference_docs
 
    print('keyword: ', keyword)
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    print('modified keyword: ', keyword)
    
    top_k = numberOfDocs
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID"   # SEMANTIC
            }},
        )
        
        docs = retriever.invoke(keyword)
        # print('docs: ', docs)
        print('--> docs from knowledge base')
        for i, doc in enumerate(docs):
            # print_doc(i, doc)
            
            content = ""
            if doc.page_content:
                content = doc.page_content
            
            score = doc.metadata["score"]
            
            link = ""
            if "s3Location" in doc.metadata["location"]:
                link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""
                
                # print('link:', link)    
                pos = link.find(f"/{doc_prefix}")
                name = link[pos+len(doc_prefix)+1:]
                encoded_name = parse.quote(name)
                # print('name:', name)
                link = f"{path}{doc_prefix}{encoded_name}"
                
            elif "webLocation" in doc.metadata["location"]:
                link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
                name = "WEB"

            url = link
            # print('url:', url)
            
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'score': score,
                        'url': url,
                        'from': 'RAG'
                    },
                )
            )    
    
    # grading        
    filtered_docs = grade_documents(keyword, relevant_docs)

    filtered_docs = check_duplication(filtered_docs) # duplication checker

    relevant_context = ""
    for i, document in enumerate(filtered_docs):
        print(f"{i}: {document}")
        if document.page_content:
            relevant_context += document.page_content + "\n\n"        
    print('relevant_context: ', relevant_context)
    
    if len(filtered_docs):
        reference_docs += filtered_docs
        return relevant_context
    else:        
        no_content_msg = "No relevant documents found."
        print(f"--> {no_content_msg}")
        return no_content_msg
    
@tool
def search_by_tavily(keyword: str) -> str:
    """
    Search general knowledge by keyword and then return the result as a string.
    keyword: search keyword
    return: the information of keyword
    """    
    global reference_docs    
    answer = ""
    
    if tavily_key:
        keyword = keyword.replace('\'','')
        
        search = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=True,
            api_wrapper=tavily_api_wrapper,
            search_depth="advanced", # "basic"
            # include_domains=["google.com", "naver.com"]
        )
                    
        try: 
            output = search.invoke(keyword)
            print('tavily output: ', output)
            
            for result in output:
                print('result: ', result)
                if result:
                    content = result.get("content")
                    url = result.get("url")
                    
                    reference_docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                'name': 'WWW',
                                'url': url,
                                'from': 'tavily'
                            },
                        )
                    )                
                    answer = answer + f"{content}, URL: {url}\n"
        
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)           
            # raise Exception ("Not able to request to tavily")   
        
    return answer

tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_knowledge_base]        

####################### LangGraph #######################
# Chat Agent Executor
#########################################################
def run_agent_executor(query, st, debugMode):
    chatModel = get_chat()     
    model = chatModel.bind_tools(tools)

    class State(TypedDict):
        # messages: Annotated[Sequence[BaseMessage], operator.add]
        messages: Annotated[list, add_messages]

    tool_node = ToolNode(tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        print("###### should_continue ######")

        print('state: ', state)
        messages = state["messages"]    

        last_message = messages[-1]
        print('last_message: ', last_message)
        
        # print('last_message: ', last_message)
        
        # if not isinstance(last_message, ToolMessage):
        #     return "end"
        # else:                
        #     return "continue"
        if isinstance(last_message, ToolMessage) or last_message.tool_calls:
            print(f"tool_calls: ", last_message.tool_calls)

            for message in last_message.tool_calls:
                print(f"tool name: {message['name']}, args: {message['args']}")
                # update_state_message(f"calling... {message['name']}", config)

            print(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"
        
        #if not last_message.tool_calls:
        else:
            print("Final: ", last_message.content)
            print("--- END ---")
            return "end"
           
    def call_model(state: State, config):
        print("###### call_model ######")
        print('state: ', state["messages"])
                
        if isKorean(state["messages"][0].content)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
                "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            )
        else: 
            system = (            
                "You are a conversational AI designed to answer in a friendly way to a question."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            )

        for attempt in range(3):   
            print('attempt: ', attempt)
            try:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                chain = prompt | model
                    
                response = chain.invoke(state["messages"])
                print('call_model response: ', response)

                if isinstance(response.content, list):            
                    for re in response.content:
                        if "type" in re:
                            if re['type'] == 'text':
                                print(f"--> {re['type']}: {re['text']}")

                                status = re['text']
                                print('status: ',status)
                                
                                status = status.replace('`','')
                                status = status.replace('\"','')
                                status = status.replace("\'",'')
                                
                                print('status: ',status)
                                if status.find('<thinking>') != -1:
                                    print('Remove <thinking> tag.')
                                    status = status[status.find('<thinking>')+11:status.find('</thinking>')]
                                    print('status without tag: ', status)

                                if debugMode=="Debug":
                                    st.info(status)
                                
                            elif re['type'] == 'tool_use':                
                                print(f"--> {re['type']}: {re['name']}, {re['input']}")

                                if debugMode=="Debug":
                                    st.info(f"{re['type']}: {re['name']}, {re['input']}")
                            else:
                                print(re)
                        else: # answer
                            print(response.content)
                break
            except Exception:
                response = AIMessage(content="답변을 찾지 못하였습니다.")

                err_msg = traceback.format_exc()
                print('error message: ', err_msg)
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
    print("msg: ", msg)

    for i, doc in enumerate(reference_docs):
        print(f"--> {i}: {doc}")
        
    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)

    msg = extract_thinking_tag(msg, st, debugMode)
    
    return msg+reference

####################### LangGraph #######################
# Agentic Workflow: Tool Use (partial tool을 활용)
#########################################################

def run_agent_executor2(query, st, debugMode):        
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        answer: str

    tool_node = ToolNode(tools)
            
    def create_agent(chat, tools):        
        tool_names = ", ".join([tool.name for tool in tools])
        print("tool_names: ", tool_names)

        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."

            "Use the provided tools to progress towards answering the question."
            "If you are unable to fully answer, that's OK, another assistant with different tools "
            "will help where you left off. Execute what you can to make progress."
            "You have access to the following tools: {tool_names}."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        prompt = prompt.partial(tool_names=tool_names)
        
        return prompt | chat.bind_tools(tools)
    
    def agent_node(state, agent, name):
        print(f"###### agent_node:{name} ######")

        last_message = state["messages"][-1]
        print('last_message: ', last_message)
        if isinstance(last_message, ToolMessage) and last_message.content=="":    
            print('last_message is empty') 
            answer = get_basic_answer(state["messages"][0].content)  
            return {
                "messages": [AIMessage(content=answer)],
                "answer": answer
            }
        
        response = agent.invoke(state["messages"])
        print('response: ', response)

        if "answer" in state:
            answer = state['answer']
        else:
            answer = ""

        if isinstance(response.content, list):      
            for re in response.content:
                if "type" in re:
                    if re['type'] == 'text':
                        print(f"--> {re['type']}: {re['text']}")

                        status = re['text']
                        if status.find('<thinking>') != -1:
                            print('Remove <thinking> tag.')
                            status = status[status.find('<thinking>')+11:status.find('</thinking>')]
                            print('status without tag: ', status)

                        if debugMode=="Debug":
                            st.info(status)

                    elif re['type'] == 'tool_use':                
                        print(f"--> {re['type']}: name: {re['name']}, input: {re['input']}")

                        if debugMode=="Debug":
                            st.info(f"{re['type']}: name: {re['name']}, input: {re['input']}")
                    else:
                        print(re)
                else: # answer
                    answer += '\n'+response.content
                    print(response.content)
                    break

        response = AIMessage(**response.dict(exclude={"type", "name"}), name=name)     
        print('message: ', response)
        
        return {
            "messages": [response],
            "answer": answer
        }
    
    def final_answer(state):
        print(f"###### final_answer ######")        

        answer = ""        
        if "answer" in state:
            answer = state['answer']            
        else:
            answer = state["messages"][-1].content

        if answer.find('<thinking>') != -1:
            print('Remove <thinking> tag.')
            answer = answer[answer.find('</thinking>')+12:]
        print('answer: ', answer)
        
        return {
            "answer": answer
        }
    
    chat = get_chat()
    
    execution_agent = create_agent(chat, tools)
    
    execution_agent_node = functools.partial(agent_node, agent=execution_agent, name="execution_agent")
    
    def should_continue(state: State, config) -> Literal["continue", "end"]:
        print("###### should_continue ######")
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
        
        last_message = messages[-1]        
        if not last_message.tool_calls:
            print("Final: ", last_message.content)
            print("--- END ---")
            return "end"
        else:      
            print(f"tool_calls: ", last_message.tool_calls)

            for message in last_message.tool_calls:
                print(f"tool name: {message['name']}, args: {message['args']}")
                # update_state_message(f"calling... {message['name']}", config)

            print(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"

    def buildAgentExecutor():
        workflow = StateGraph(State)

        workflow.add_node("agent", execution_agent_node)
        workflow.add_node("action", tool_node)
        workflow.add_node("final_answer", final_answer)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": "final_answer",
            },
        )
        workflow.add_edge("action", "agent")
        workflow.add_edge("final_answer", END)

        return workflow.compile()

    app = buildAgentExecutor()
            
    inputs = [HumanMessage(content=query)]
    config = {"recursion_limit": 50}
    
    msg = ""
    # for event in app.stream({"messages": inputs}, config, stream_mode="values"):   
    #     # print('event: ', event)
        
    #     if "answer" in event:
    #         msg = event["answer"]
    #     else:
    #         msg = event["messages"][-1].content
    #     # print('message: ', message)

    output = app.invoke({"messages": inputs}, config)
    print('output: ', output)

    msg = output['answer']

    return msg

def get_basic_answer(query):
    print('#### get_basic_answer ####')
    chat = get_chat()

    if isKorean(query)==True:
        system = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "You will be acting as a thoughtful advisor."
            "Using the following conversation, answer friendly for the newest question." 
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."     
        )    
    
    human = "Question: {input}"    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        ("human", human)
    ])    
    
    chain = prompt | chat    
    output = chain.invoke({"input": query})
    print('output.content: ', output.content)

    return output.content


####################### LangGraph #######################
# Agentic Workflow: Reflection 
#########################################################

def init_enhanced_search(st, debugMode):
    chat = get_chat() 

    model = chat.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tool_node = ToolNode(tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
            
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:                
            return "continue"

    def call_model(state: State, config):
        print('##### call_model #####')

        messages = state["messages"]
        # print('messages: ', messages)

        last_message = messages[-1]
        print('last_message: ', last_message)

        if isinstance(last_message, ToolMessage) and last_message.content=="":              
            print('last_message is empty')      
            print('question: ', state["messages"][0].content)
            answer = get_basic_answer(state["messages"][0].content)          
            return {"messages": [AIMessage(content=answer)]}
            
        if isKorean(messages[0].content)==True:
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
        print('call_model response: ', response)
              
        # state messag
        if response.tool_calls:
            print('tool_calls response: ', response.tool_calls)

            toolinfo = response.tool_calls[-1]            
            if toolinfo['type'] == 'tool_call':
                print('tool name: ', toolinfo['name'])         

            if debugMode=="Debug":
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

def enhanced_search(query, config, st, debugMode):
    print("###### enhanced_search ######")
    inputs = [HumanMessage(content=query)]

    app_enhanced_search = init_enhanced_search(st, debugMode)        
    result = app_enhanced_search.invoke({"messages": inputs}, config)   
    print('result: ', result)
            
    message = result["messages"][-1]
    print('enhanced_search: ', message)

    if message.content.find('<result>')==-1:
        return message.content
    else:
        return message.content[message.content.find('<result>')+8:message.content.find('</result>')]
    
def run_knowledge_guru(query, st, debugMode):
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        reflection: list
        search_queries: list
            
    def generate(state: State, config):    
        print("###### generate ######")
        print('state: ', state["messages"])
        print('task: ', state['messages'][0].content)

        if debugMode=="Debug":
            st.info(f"검색을 수행합니다. 검색어: {state['messages'][0].content}")
        
        draft = enhanced_search(state['messages'][0].content, config, st, debugMode)  
        print('draft: ', draft)

        if debugMode=="Debug":
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
        print("###### reflect ######")
        print('state: ', state["messages"])    
        print('draft: ', state["messages"][-1].content)
        
        if debugMode=="Debug":
            st.info('초안을 검토하여 부족하거나 보강할 내용을 찾고, 추가 검색어를 추출합니다.')

        reflection = []
        search_queries = []
        for attempt in range(20):
            chat = get_chat()
            if isKorean(state["messages"][-1].content):
                structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
            else:
                structured_llm = chat.with_structured_output(Research, include_raw=True)
            
            info = structured_llm.invoke(state["messages"][-1].content)
            print(f'attempt: {attempt}, info: {info}')
                
            if not info['parsed'] == None:
                parsed_info = info['parsed']
                # print('reflection: ', parsed_info.reflection)                
                reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                search_queries = parsed_info.search_queries
                
                print('reflection: ', parsed_info.reflection)            
                print('search_queries: ', search_queries)      

                if debugMode=="Debug":  
                    st.info(f'개선할 사항: {parsed_info.reflection}')
                    st.info(f'추가 검색어: {search_queries}')        
                break
        
        return {
            "messages": state["messages"],
            "reflection": reflection,
            "search_queries": search_queries
        }

    def revise_answer(state: State, config):   
        print("###### revise_answer ######")
        
        if debugMode=="Debug":
            st.info("개선할 사항을 반영하여 답변을 생성중입니다.")
        human = (
            "Revise your previous answer using the new information."
            "You should use the previous critique to add important information to your answer." 
            "provide the final answer with <result> tag."

            "critique:"
            "{reflection}"

            "information:"
            "{content}"
        )
                    
        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                ("human", human),
            ]
        )
            
        content = []        
        if useEnhancedSearch: # search agent
            for q in state["search_queries"]:
                response = enhanced_search(q, config)
                # print(f'q: {q}, response: {response}')
                content.append(response)                   
        else:
            search = TavilySearchResults(
                max_results=2,
                include_answer=True,
                include_raw_content=True,
                api_wrapper=tavily_api_wrapper,
                search_depth="advanced", 
                # include_domains=["google.com", "naver.com"]
            )
            for q in state["search_queries"]:
                response = search.invoke(q)     
                for r in response:
                    if 'content' in r:
                        content.append(r['content'])     

        chat = get_chat()
        reflect = reflection_prompt | chat
            
        messages = state["messages"]
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [messages[0]] + [
            cls_map[msg.type](content=msg.content) for msg in messages[1:]
        ]
        print('translated: ', translated)     
           
        res = reflect.invoke(
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
        print('revised_answer: ', response.content)
                
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "messages": [response], 
            "revision_number": revision_number + 1
        }
    
    MAX_REVISIONS = 1
    def should_continue(state: State, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
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
        "parallel_processing": parallel_processing
    }
    
    for output in app.stream({"messages": inputs}, config):   
        for key, value in output.items():
            print(f"Finished: {key}")
            #print("value: ", value)
            
    print('value: ', value)
        
    msg = value["messages"][-1].content

    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)

    return msg+reference


####################### LangGraph #######################
# Agentic Workflow: Planning (Advanced CoT)
#########################################################
def run_planning(query, st, debugMode):
    class State(TypedDict):
        input: str
        plan: list[str]
        past_steps: Annotated[List[Tuple], operator.add]
        info: Annotated[List[Tuple], operator.add]
        answer: str

    def plan_node(state: State, config):
        print("###### plan ######")
        print('input: ', state["input"])

        if debugMode=="Debug":
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
        chat = get_chat()
        planner = planner_prompt | chat
        response = planner.invoke({
            "question": state["input"]
        })
        print('response.content: ', response.content)
        result = response.content
        
        #output = result[result.find('<result>')+8:result.find('</result>')]
        output = result
        
        plan = output.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')
        print('planning_steps: ', planning_steps)

        if debugMode=="Debug":
            st.info(f"생성된 계획: {planning_steps}")
        
        return {
            "input": state["input"],
            "plan": planning_steps
        }
    
    def generate_answer(chat, relevant_docs, text):    
        relevant_context = ""
        for document in relevant_docs:
            relevant_context = relevant_context + document.page_content + "\n\n"        
        # print('relevant_context: ', relevant_context)

        if debugMode=="Debug":
            st.info(f"계획을 수행합니다. 현재 계획 {text}")

        # generating
        if isKorean(text)==True:
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
                        
        chain = prompt | chat
        
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
        print("###### execute ######")
        print('input: ', state["input"])
        plan = state["plan"]
        print('plan: ', plan) 
        
        chat = get_chat()

        if debugMode=="Debug":
            st.info(f"검색을 수행합니다. 검색어 {plan[0]}")
        
        # retrieve
        relevant_docs = retrieve_documents_from_knowledge_base(plan[0], top_k=4)
        relevant_docs += retrieve_documents_from_tavily(plan[0], top_k=4)
            
        # grade
        filtered_docs = grade_documents(plan[0], relevant_docs) # grading    
        filtered_docs = check_duplication(filtered_docs) # check duplication
                
        # generate
        result = generate_answer(chat, relevant_docs, plan[0])
        
        print('task: ', plan[0])
        print('executor output: ', result)

        global reference_docs
        if len(filtered_docs):
            reference_docs += filtered_docs

        if debugMode=="Debug":
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
        print('#### replan ####')
        print('state of replan node: ', state)

        if len(state["plan"]) == 1:
            print('last plan: ', state["plan"])
            return {"response":state["info"][-1]}
        
        if debugMode=="Debug":
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

            "완료한 단계는 아래와 같습니다."
            "Past steps:"
            "{past_steps}"
            
            "당신은 Original Plan의 원래 계획을 상황에 맞게 수정하세요."
            "계획에 아직 해야 할 단계만 추가하세요. 이전에 완료한 단계는 계획에 포함하지 마세요."                
            "수정된 계획에는 <plan> tag를 붙여주세요."
            "만약 더 이상 계획을 세우지 않아도 Question의 주어진 질문에 답변할 있다면, 최종 결과로 Question에 대한 답변을 <result> tag를 붙여 전달합니다."
            
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
        
        chat = get_chat()
        replanner = replanner_prompt | chat
        
        response = replanner.invoke({
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": state["past_steps"]
        })
        print('replanner output: ', response.content)
        result = response.content

        if result.find('<plan>') == -1:
            return {"response":response.content}
        else:
            output = result[result.find('<plan>')+6:result.find('</plan>')]
            print('plan output: ', output)

            plans = output.strip().replace('\n\n', '\n')
            planning_steps = plans.split('\n')
            print('planning_steps: ', planning_steps)

            if debugMode=="Debug":
                st.info(f"새로운 계획: {planning_steps}")

            return {"plan": planning_steps}
        
    def should_end(state: State) -> Literal["continue", "end"]:
        print('#### should_end ####')
        # print('state: ', state)
        
        if "response" in state and state["response"]:
            print('response: ', state["response"])            
            next = "end"
        else:
            print('plan: ', state["plan"])
            next = "continue"
        print(f"should_end response: {next}")
        
        return next
        
    def final_answer(state: State) -> str:
        print('#### final_answer ####')
        
        # get final answer
        context = "".join(f"{info}\n" for info in state['info'])
        print('context: ', context)
        
        query = state['input']
        print('query: ', query)

        if debugMode=="Debug":
            st.info(f"최종 답변을 생성합니다.")
        
        if isKorean(query)==True:
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
                    
        chat = get_chat()
        chain = prompt | chat
        
        try: 
            response = chain.invoke(
                {
                    "context": context,
                    "input": query,
                }
            )
            result = response.content

            if result.find('<result>')==-1:
                output = result
            else:
                output = result[result.find('<result>')+8:result.find('</result>')]
                
            print('output: ', output)
            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)      
            
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
            print(f"Finished: {key}")
            #print("value: ", value)            
    print('value: ', value)

    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)
    
    return value["answer"]+reference

    
####################### LangGraph #######################
# Agentic Workflow Multi-agent Collaboration 
#########################################################
def run_long_form_writing_agent(query, st, debugMode):
    # Workflow - Reflection
    class ReflectionState(TypedDict):
        draft : str
        reflection : List[str]
        search_queries : List[str]
        revised_draft: str
        revision_number: int
        reference: List[str]
        
    class Reflection(BaseModel):
        missing: str = Field(description="Critique of what is missing.")
        advisable: str = Field(description="Critique of what is helpful for better writing")
        superfluous: str = Field(description="Critique of what is superfluous")

    class Research(BaseModel):
        """Provide reflection and then follow up with search queries to improve the writing."""

        reflection: Reflection = Field(description="Your reflection on the initial writing.")
        search_queries: list[str] = Field(
            description="1-3 search queries for researching improvements to address the critique of your current writing."
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
        
    def reflect_node(state: ReflectionState, config):
        print("###### reflect ######")
        draft = state['draft']
        
        idx = config.get("configurable", {}).get("idx")
        print('reflect_node idx: ', idx)

        if debugMode=="Debug":
            st.info(f"{idx}: draft에서 개선 사항을 도출합니다.")
    
        reflection = []
        search_queries = []
        for attempt in range(20):
            chat = get_chat()
            if isKorean(draft):
                structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
            else:
                structured_llm = chat.with_structured_output(Research, include_raw=True)
            
            try:
                # print('draft: ', draft)
                info = structured_llm.invoke(draft)
                print(f'attempt: {attempt}, info: {info}')
                    
                if not info['parsed'] == None:
                    parsed_info = info['parsed']
                    # print('reflection: ', parsed_info.reflection)                
                    reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
                    search_queries = parsed_info.search_queries

                    if debugMode=="Debug":
                        st.info(f"{idx}: 개선사항: {reflection}")
                    
                    print('reflection: ', parsed_info.reflection)
                    print('search_queries: ', search_queries)
            
                    if isKorean(draft):
                        translated_search = []
                        for q in search_queries:
                            chat = get_chat()
                            if isKorean(q):
                                search = traslation(chat, q, "Korean", "English")
                            else:
                                search = traslation(chat, q, "English", "Korean")
                            translated_search.append(search)
                            
                        print('translated_search: ', translated_search)
                        search_queries += translated_search

                    if debugMode=="Debug":
                        st.info(f"검색어: {search_queries}")

                    print('search_queries (mixed): ', search_queries)
                    break
            except Exception:
                print('---> parsing error')

                err_msg = traceback.format_exc()
                print('error message: ', err_msg)
                # raise Exception ("Not able to request to LLM")
            
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        return {
            "reflection": reflection,
            "search_queries": search_queries,
            "revision_number": revision_number + 1
        }

    def retrieve_for_writing(conn, q, config):
        idx = config.get("configurable", {}).get("idx") 
         
        if debugMode=="Debug":
            st.info(f"검색을 수행합니다. 검색어: {q}")

        relevant_docs = retrieve_documents_from_knowledge_base(q, top_k=numberOfDocs)
        relevant_docs += retrieve_documents_from_tavily(q, top_k=numberOfDocs)

        # translate
        # docs = []
        # for doc in relevant_docs:
        #     chat = get_chat()
        #     if not isKorean(doc.page_content):
        #         translated_content = traslation(chat, doc.page_content, "English", "Korean")
        #         doc.page_content = translated_content
        #     docs.append(doc)
        # print('translated relevant docs: ', docs)
                
        # grade
        filtered_docs = grade_documents(q, relevant_docs) # grading    
        filtered_docs = check_duplication(filtered_docs) # check duplication
                                
        conn.send(filtered_docs)
        conn.close()

    def parallel_retriever(search_queries, config):
        relevant_documents = []    
        
        processes = []
        parent_connections = []
        for q in search_queries:
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
                
            process = Process(target=retrieve_for_writing, args=(child_conn, q, config))
            processes.append(process)

        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            rel_docs = parent_conn.recv()

            if(len(rel_docs)>=1):
                for doc in rel_docs:
                    relevant_documents.append(doc)    

        for process in processes:
            process.join()
        
        #print('relevant_docs: ', relevant_docs)
        return relevant_documents

    def retrieve_docs(search_queries, config):
        docs = []
        
        parallel_retrieval = config.get("configurable", {}).get("parallel_retrieval")
        print('parallel_retrieval: ', parallel_retrieval)
        
        if parallel_retrieval == 'enable':
            docs = parallel_retriever(search_queries, config)
        else:
            for q in search_queries:      
                if debugMode=="Debug":
                    st.info(f"검색을 수행합니다. 검색어: {q}")

                relevant_docs = retrieve_documents_from_knowledge_base(q, top_k=numberOfDocs)
                relevant_docs += retrieve_documents_from_tavily(q, top_k=numberOfDocs)
            
                # grade
                docs += grade_documents(q, relevant_docs) # grading
                    
            docs = check_duplication(docs) # check duplication
            for i, doc in enumerate(docs):
                print(f"#### {i}: {doc.page_content[:100]}")
                            
        return docs
        
    def revise_draft(state: ReflectionState, config):   
        print("###### revise_draft ######")
        
        draft = state['draft']
        search_queries = state['search_queries']
        reflection = state['reflection']
        print('draft: ', draft)
        print('search_queries: ', search_queries)
        print('reflection: ', reflection)

        idx = config.get("configurable", {}).get("idx")
        print('revise_draft idx: ', idx)
        
        if debugMode=="Debug":
            st.info(f"{idx}: 개선사항을 반영하여 새로운 답변을 생성합니다.")
        
        # reference = state['reference'] if 'reference' in state else []     
        if 'reference' in state:
            reference = state['reference'] 
        else:
            reference = []            

        if len(search_queries) and len(reflection):
            docs = retrieve_docs(search_queries, config)        
            print('docs: ', docs)
                    
            content = []   
            if len(docs):                
                for d in docs:
                    content.append(d.page_content)            
                print('content: ', content)
                                    
                if isKorean(draft):
                    system = (
                        "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
                        "draft을 critique과 information 사용하여 수정하십시오."
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
                            
                revise_prompt = ChatPromptTemplate([
                    ('system', system),
                    ('human', human)
                ])

                chat = get_chat()
                reflect = revise_prompt | chat
                
                res = reflect.invoke(
                    {
                        "draft": draft,
                        "reflection": reflection,
                        "content": content
                    }
                )
                output = res.content
                # print('output: ', output)
                
                if output.find('<result>') == -1:
                    revised_draft = output
                else:
                    revised_draft = output[output.find('<result>')+8:output.find('</result>')]
                    
                # print('--> draft: ', draft)
                print('--> reflection: ', reflection)
                print('--> revised_draft: ', revised_draft)

                st.info(f"revised_draft: {revised_draft}")

                reference += docs
                print('len(reference): ', len(reference))
            else:
                print('No relevant document!')
                revised_draft = draft
        else:
            print('No reflection!')
            revised_draft = draft
            
        revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        
        return {
            "revised_draft": revised_draft,            
            "revision_number": revision_number,
            "reference": reference
        }
        
    MAX_REVISIONS = 1
    def should_continue(state: ReflectionState, config):
        print("###### should_continue ######")
        max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
        print("max_revisions: ", max_revisions)
            
        if state["revision_number"] > max_revisions:
            return "end"
        return "continue"        
    
    def buildReflection():
        workflow = StateGraph(ReflectionState)

        # Add nodes
        workflow.add_node("reflect_node", reflect_node)
        workflow.add_node("revise_draft", revise_draft)

        # Set entry point
        workflow.set_entry_point("reflect_node")
        
        workflow.add_conditional_edges(
            "revise_draft", 
            should_continue, 
            {
                "end": END, 
                "continue": "reflect_node"
            }
        )

        # Add edges
        workflow.add_edge("reflect_node", "revise_draft")
        
        return workflow.compile()
    
    # Workflow - Long Writing
    class State(TypedDict):
        instruction : str
        planning_steps : List[str]
        drafts : List[str]
        # num_steps : int
        final_doc : str
        word_count : int
            
    def plan_node(state: State, config):
        print("###### plan ######")
        instruction = state["instruction"]
        print('subject: ', instruction)

        if debugMode=="Debug":
            st.info(f"계획을 생성합니다. 요청사항: {instruction}")
        
        if isKorean(instruction):
            system = (
                "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."
                "당신은 글쓰기 지시 사항을 여러 개의 하위 작업으로 나눌 것입니다."
                "글쓰기 계획은 5단계 이하로 작성합니다."
                "각 하위 작업은 에세이의 한 단락 작성을 안내할 것이며, 해당 단락의 주요 내용과 단어 수 요구 사항을 포함해야 합니다."
                "각 하위 작업이 명확하고 구체적인지, 그리고 모든 하위 작업이 작문 지시 사항의 전체 내용을 다루고 있는지 확인하세요."
                "과제를 너무 세분화하지 마세요. 각 하위 과제의 문단은 500단어 이상 3000단어 이하여야 합니다."
                "다른 내용은 출력하지 마십시오. 이것은 진행 중인 작업이므로 열린 결론이나 다른 수사학적 표현을 생략하십시오."     
            )
            human = (
                "글쓰기 지시 사항은 아래와 같습니다."
                "Instruction:"
                "{instruction}"
                
                "다음 형식으로 나누어 주시기 바랍니다. 각 하위 작업은 한 줄을 차지합니다:"
                "1. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [Word count requirement, e.g., 800 words]"
                "2. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [word count requirement, e.g. 1500 words]."
                "..."                                       
            )
        else:
            system = (
                "You are a helpful assistant highly skilled in long-form writing."
                "You will break down the writing instruction into multiple subtasks."
                "Writing plans are created in five steps or less."
                "Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph."
                "Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction."
                "Do not split the subtasks too finely; each subtask's paragraph should be no less than 500 words and no more than 3000 words."
                "Do not output any other content. As this is an ongoing work, omit open-ended conclusions or other rhetorical hooks."
            )
            human = (                
                "The writing instruction is as follows:"
                "<instruction>"
                "{instruction}"
                "<instruction>"
                
                "Please break it down in the following format, with each subtask taking up one line:"
                "1. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [Word count requirement, e.g., 800 words]"
                "2. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [word count requirement, e.g. 1500 words]."
                "..."                
            )
        
        planner_prompt = ChatPromptTemplate([
            ('system', system),
            ('human', human) 
        ])
                
        chat = get_chat()
        
        planner = planner_prompt | chat
    
        response = planner.invoke({"instruction": instruction})
        print('response: ', response.content)
    
        plan = response.content.strip().replace('\n\n', '\n')
        planning_steps = plan.split('\n')        
        print('planning_steps: ', planning_steps)

        if debugMode=="Debug":
            st.info(f"생성된 계획: {planning_steps}")
            
        return {
            "instruction": instruction,
            "planning_steps": planning_steps
        }
        
    def execute_node(state: State, config):
        print("###### execute_node ######")        
        
        instruction = state["instruction"]        
        if isKorean(instruction):
            system = (
                "당신은 훌륭한 글쓰기 도우미입니다." 
                "당신은 글쓰기 plan에 따라 instruction에 대한 글을 작성하고자 합니다."
                "이전 단계에서 written text까지 작성하였고, next step을 계속 작성합니다."
                "글이 끊어지지 않고 잘 이해되도록 하나의 문단을 충분히 길게 작성합니다."
                
                "필요하다면 앞에 작은 부제를 추가할 수 있습니다."
                "이미 작성된 텍스트를 반복하지 말고 작성한 문단만 출력하세요."                
                "Markdown 포맷으로 서식을 작성하세요."                
            )
            human = (
                "아래는 이전 단계에서 작성된 텍스트입니다."
                "Written text:"
                "{text}"

                "글쓰기 지시사항은 아래와 같습니다."                
                "Instruction:"
                "{intruction}"

                "전체 글쓰기 단계는 아래와 같습니다."
                "Plan:"
                "{plan}"
               
                "다음으로 작성할 글쓰기 단계입니다. Instruction, plan, written text을 참조하여 next step을 계속 작성합니다."
                "Next step:"
                "{step}"
            )
        else:    
            system = (
                "You are an excellent writing assistant." 
                "You intend to write an article about instructions according to a writing plan. "
                "In the previous step, we wrote up to the written text, and we will continue writing the next step. "
                "Please help me continue writing the next paragraph based on the writing instruction, writing steps, and the already written text."

                "If needed, you can add a small subtitle at the beginning."
                "Remember to only output the paragraph you write, without repeating the already written text."
                "Use markdown syntax to format your output:"
                "- Headings: # for main, ## for sections, ### for subsections, etc."
                "- Lists: * or - for bulleted, 1. 2. 3. for numbered"
                "- Do not repeat yourself"
                "Provide the final answer with <result> tag."
            )
            human = (
                "The text written in the previous step is as follows."
                "Written text:"
                "{text}"  

                "The writing instructions are as follows."
                "Instruction:"
                "{intruction}"

                "The entire writing plan is as follows."
                "Plan:"
                "{plan}"

                "The next step to write is as follows. Please refer to the writing instruction, writing steps, and the already written text."
                "Next step:"
                "{step}"                                          
            )

        write_prompt = ChatPromptTemplate([
            ('system', system),
            ('human', human)
        ])
        
        planning_steps = state["planning_steps"]        
        if len(planning_steps) > 50:
            print("plan is too long")
            # print(plan)
            return
        
        text = ""
        drafts = []
        for idx, step in enumerate(planning_steps):            
            # Invoke the write_chain
            chat = get_chat()
            write_chain = write_prompt | chat     

            plan = ""
            for p in planning_steps:
                plan += p + '\n'

            result = write_chain.invoke({
                "intruction": instruction,
                "plan": plan,
                "text": text,
                "step": step
            })
            output = result.content
            # print('output: ', output)

            if debugMode=="Debug":
                st.info(f"수행단계: {step}")
            
            if output.find('<result>')==-1:
                draft = output
            else:
                draft = output[output.find('<result>')+8:output.find('</result>')]

            if debugMode=="Debug":
                st.info(f"생성결과: {draft}")
                                              
            print(f"--> step: {step}")
            print(f"--> draft: {draft}")
                
            drafts.append(draft)
            text += draft + '\n\n'

        return {
            "instruction": instruction,
            "drafts": drafts
        }

    def reflect_draft(conn, reflection_app, idx, config, draft):     
        inputs = {
            "draft": draft
        }     

        try:        
            output = reflection_app.invoke(inputs, config)
            print('idx: ', idx)

        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            # raise Exception ("Not able to request to LLM")    
    
        result = {
            "revised_draft": output['revised_draft'],
            "idx": idx,
            "reference": output['reference']
        }
            
        conn.send(result)    
        conn.close()
        
    def reflect_drafts_using_parallel_processing(drafts, config):
        revised_drafts = drafts
        
        processes = []
        parent_connections = []
        references = []
        
        reflection_app = buildReflection()
        
        for idx, draft in enumerate(drafts):
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)
            
            print(f"idx:{idx} --> draft:{draft}")
            
            app_config = {
                "recursion_limit": 50,
                "max_revisions": MAX_REVISIONS,
                "idx": idx,
                "parallel_retrieval": "enable"
            }
            process = Process(target=reflect_draft, args=(child_conn, reflection_app, idx, app_config, draft))
            processes.append(process)
            
        for process in processes:
            process.start()
                
        for parent_conn in parent_connections:
            result = parent_conn.recv()

            if result is not None:
                print('result: ', result)
                revised_drafts[result['idx']] = result['revised_draft']

                if result['reference']:
                    references += result['reference']

        for process in processes:
            process.join()
                
        final_doc = ""   
        for revised_draft in revised_drafts:
            final_doc += revised_draft + '\n\n'
        
        return final_doc, references

    def get_subject(query):
        system = (
            "Extract the subject of the question in 6 words or fewer."
        )
        
        human = "<question>{question}</question>"
        
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        # print('prompt: ', prompt)
        
        chat = get_chat()
        chain = prompt | chat    
        try: 
            result = chain.invoke(
                {
                    "question": query
                }
            )        
            subject = result.content
            # print('the subject of query: ', subject)
            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            raise Exception ("Not able to request to LLM")        
        return subject
    
    def markdown_to_html(body):
        html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <md-block>
        </md-block>
        <script type="module" src="https://md-block.verou.me/md-block.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.css" integrity="sha512-n5zPz6LZB0QV1eraRj4OOxRbsV7a12eAGfFcrJ4bBFxxAwwYDp542z5M0w24tKPEhKk2QzjjIpR5hpOjJtGGoA==" crossorigin="anonymous" referrerpolicy="no-referrer"/>
    </head>
    <body>
        <div class="markdown-body">
            <md-block>{body}
            </md-block>
        </div>
    </body>
    </html>"""        
        return html

    def get_references_for_markdown(docs):
        reference = ""
        nameList = []
        cnt = 1
        for i, doc in enumerate(docs):
            print(f"reference {i}: doc")
            page = ""
            if "page" in doc.metadata:
                page = doc.metadata['page']
                #print('page: ', page)            
            url = ""
            if "url" in doc.metadata:
                url = doc.metadata['url']
                #print('url: ', url)                
            name = ""
            if "name" in doc.metadata:
                name = doc.metadata['name']
                #print('name: ', name)     
            pos = name.rfind('/')
            name = name[pos+1:]
            print(f"name: {name}")
            
            excerpt = ""+doc.page_content

            excerpt = re.sub('"', '', excerpt)
            print('length: ', len(excerpt))
            
            if name in nameList:
                print('duplicated!')
            else:
                reference = reference + f"{cnt}. [{name}]({url})"
                nameList.append(name)
                cnt = cnt+1
                
        return reference
    
    def revise_answer(state: State, config):
        print("###### revise ######")
        drafts = state["drafts"]        
        print('drafts: ', drafts)
        
        parallel_revise = config.get("configurable", {}).get("parallel_revise", "enable")
        print('parallel_revise: ', parallel_revise)

        if debugMode=="Debug":
            st.info("문서를 개선합니다.")
        
        # reflection
        if parallel_revise == 'enable':  # parallel processing
            final_doc, references = reflect_drafts_using_parallel_processing(drafts, config)
        else:
            reflection_app = buildReflection()
                
            final_doc = ""   
            references = []
                        
            for idx, draft in enumerate(drafts):
                inputs = {
                    "draft": draft
                }                    
                app_config = {
                    "recursion_limit": 50,
                    "max_revisions": MAX_REVISIONS,
                    "idx": idx,
                    "parallel_retrieval": "disable"
                }
                output = reflection_app.invoke(inputs, config=app_config)
                final_doc += output['revised_draft'] + '\n\n'
                references += output['reference']

        subject = get_subject(state['instruction'])
        subject = subject.replace(" ","_")
        subject = subject.replace("?","")
        subject = subject.replace("!","")
        subject = subject.replace(".","")
        subject = subject.replace(":","")
        
        print('len(references): ', len(references))
        
        # markdown file
        markdown_key = 'markdown/'+f"{subject}.md"
        # print('markdown_key: ', markdown_key)
        
        final_doc = f"## {state['instruction']}\n\n"+final_doc

        if references:
            print('references: ', references)

            markdown_reference = get_references(references)
            
            print('markdown_reference: ', markdown_reference)

            final_doc += markdown_reference
                
        s3_client = boto3.client('s3')  
        response = s3_client.put_object(
            Bucket=bucketName,
            Key=markdown_key,
            ContentType='text/markdown',
            Body=final_doc.encode('utf-8')
        )
        # print('response: ', response)
        
        markdown_url = f"{path}{markdown_key}"
        print('markdown_url: ', markdown_url)
        
        # html file
        html_key = 'markdown/'+f"{subject}.html"
            
        html_body = markdown_to_html(final_doc)
        print('html_body: ', html_body)
        
        s3_client = boto3.client('s3')  
        response = s3_client.put_object(
            Bucket=bucketName,
            Key=html_key,
            ContentType='text/html',
            Body=html_body
        )
        # print('response: ', response)
        
        html_url = f"{path}{html_key}"
        print('html_url: ', html_url)

        final_doc += f"\n[미리보기 링크]({html_url})\n\n[다운로드 링크 - {subject}.md]({markdown_url})"
        
        return {
            "final_doc": final_doc
        }
        
    def buildLongformWriting():
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("plan_node", plan_node)
        workflow.add_node("execute_node", execute_node)
        workflow.add_node("revise_answer", revise_answer)

        # Set entry point
        workflow.set_entry_point("plan_node")

        # Add edges
        workflow.add_edge("plan_node", "execute_node")
        workflow.add_edge("execute_node", "revise_answer")
        workflow.add_edge("revise_answer", END)
        
        return workflow.compile()
    
    # initiate
    global contentList, reference_docs
    contentList = []
    reference_docs = []
    
    # Run the workflow
    app = buildLongformWriting()    
    inputs = {
        "instruction": query
    }    
    config = {
        "recursion_limit": 100,
        "parallel_revise": parallel_processing
    }
    
    output = app.invoke(inputs, config)
    print('output: ', output)
    
    return output['final_doc']


# Test 
# draft = """### 건강한 exosome의 분비를 촉진하는 방법

# 건강한 exosome의 분비를 촉진하는 방법을 탐구하는 것은 당뇨병과 같은 대사 질환의 예방과 치료에 중요한 단서를 제공할 수 있습니다. 여기에는 식이, 운동, 그리고 다른 생활 습관이 어떻게 exosome의 분비와 기능에 영향을 미칠 수 있는지 포함됩니다. 먼저, 식이는 exosome의 분비에 중요한 역할을 합니다. 특히, 다양한 영양소와 항산화제가 풍부한 식이는 지방 조직의 건강을 유지하고, 이에 따라 건강한 exosome의 분비를 촉진할 수 있습니다. 예를 들어, 오메가-3 지방산이 풍부한 식품은 지방 조직의 염증을 감소시키고, 이에 따라 건강한 exosome의 분비를 촉진할 수 있습니다. 또한, 항산화제가 풍부한 식품, 예를 들어 베리류, 견과류, 그리고 녹차는 지방 조직의 산화 스트레스를 감소시키고, 이에 따라 건강한 exosome의 분비를 촉진할 수 있습니다.

# 운동 또한 exosome의 분비에 중요한 역할을 합니다. 규칙적인 운동은 지방 조직의 대사 활동을 증가시키고, 이에 따라 건강한 exosome의 분비를 촉진할 수 있습니다. 특히, 중강도 운동은 지방 조직의 염증을 감소시키고, 이에 따라 건강한 exosome의 분비를 촉진할 수 있습니다. 또한, 운동은 인슐린 감수성을 개선하고, 이에 따라 혈당 수치를 조절할 수 있습니다. 이는 당뇨병의 발병과 진행을 예방하는 데 중요한 역할을 할 수 있습니다.

# 다른 생활 습관 또한 exosome의 분비에 영향을 미칠 수 있습니다. 예를 들어, 충분한 수면은 지방 조직의 대사 활동을 증가시키고, 이에 따라 건강한 exosome의 분비를 촉진할 수 있습니다. 또한, 스트레스 관리는 지방 조직의 염증을 감소시키고, 이에 따라 건강한 exosome의 분비를 촉진할 수 있습니다. 스트레스는 지방 조직의 염증을 증가시키고, 이에 따라 비건강한 exosome의 분비를 촉진할 수 있습니다. 따라서, 스트레스 관리는 건강한 exosome의 분비를 촉진하는 데 중요한 역할을 할 수 있습니다.

# 이러한 방법을 통해 건강한 exosome의 분비를 촉진함으로써, 당뇨병과 같은 대사 질환의 예방과 치료에 중요한 역할을 할 수 있습니다. 예를 들어, 건강한 exosome의 분비를 촉진함으로써, 인슐린 감수성을 개선하고 염증 반응을 조절할 수 있습니다. 이는 당뇨병의 발병을 예방하고 진행을 늦출 수 있는 중요한 전략이 될 수 있습니다. 또한, 건강한 exosome의 분비를 촉진함으로써, 당뇨병의 합병증을 예방하고 치료할 수 있습니다. 이러한 연구는 현재 활발히 진행되고 있으며, 미래에는 더 많은 가능성을 열어줄 것으로 기대됩니다. 따라서, 건강한 exosome의 분비를 촉진하는 방법을 탐구함으로써, 당뇨병과 같은 질병의 예방과 치료에 중요한 역할을 할 수 있습니다."""


# def run_long_form_writing_agent2(query, st, debugMode):
#     class ReflectionState(TypedDict):
#         draft : str
#         reflection : List[str]
#         search_queries : List[str]
#         revised_draft: str
#         revision_number: int
#         references: List[str]

#     class Reflection(BaseModel):
#         missing: str = Field(description="Critique of what is missing.")
#         advisable: str = Field(description="Critique of what is helpful for better writing")
#         superfluous: str = Field(description="Critique of what is superfluous")

#     class Research(BaseModel):
#         """Provide reflection and then follow up with search queries to improve the writing."""

#         reflection: Reflection = Field(description="Your reflection on the initial writing.")
#         search_queries: list[str] = Field(
#             description="1-3 search queries for researching improvements to address the critique of your current writing."
#         )

#     class ReflectionKor(BaseModel):
#         missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
#         advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
#         superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

#     class ResearchKor(BaseModel):
#         """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""

#         reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
#         search_queries: list[str] = Field(
#             description="도출된 비평을 해결하기 위한 3개 이내의 검색어"
#         ) 
    
#     def should_continue(state: ReflectionState, config):
#         print("###### should_continue ######")
#         max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
#         print("max_revisions: ", max_revisions)
            
#         if state["revision_number"] > max_revisions:
#             return "end"
#         return "continue"        
    
#     def revise_draft(state: ReflectionState, config):   
#         print("###### revise_draft ######")
        
#         draft = state['draft']
#         search_queries = state['search_queries']
#         reflection = state['reflection']
#         print('draft: ', draft)
#         print('search_queries: ', search_queries)
#         print('reflection: ', reflection)

#         if debugMode=="Debug":
#             st.info(f"개선사항을 반영하여 새로운 답변을 생성합니다.")
                            
#         # reference = state['reference'] if 'reference' in state else []     
#         if 'references' in state:
#             references = state['references'] 
#         else:
#             references = []            
#         print('----> length of previous references: ', len(references))

#         if len(search_queries) and len(reflection):
#             docs = retrieve_docs(search_queries, config)
#             print('docs: ', docs)
                    
#             content = []   
#             if len(docs):
#                 for d in docs:
#                     content.append(d.page_content)            
#                 print('content: ', content)
                                    
#                 if isKorean(draft):
#                     system = (
#                         "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
#                         "draft을 critique과 information 사용하여 수정하십시오."
#                         "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
#                     )
#                     human = (
#                         "draft:"
#                         "{draft}"
                                    
#                         "critique:"
#                         "{reflection}"

#                         "information:"
#                         "{content}"
#                     )
#                 else:    
#                     system = (
#                         "You are an excellent writing assistant." 
#                         "Revise this draft using the critique and additional information."
#                         "Provide the final answer with <result> tag."
#                     )
#                     human = (                            
#                         "draft:"
#                         "{draft}"
                                    
#                         "critique:"
#                         "{reflection}"

#                         "information:"
#                         "{content}"
#                     )
                            
#                 revise_prompt = ChatPromptTemplate([
#                     ('system', system),
#                     ('human', human)
#                 ])

#                 chat = get_chat()
#                 reflect = revise_prompt | chat
                
#                 res = reflect.invoke(
#                     {
#                         "draft": draft,
#                         "reflection": reflection,
#                         "content": content
#                     }
#                 )
#                 output = res.content
#                 # print('output: ', output)
                
#                 if output.find('<result>') == -1:
#                     revised_draft = output
#                 else:
#                     revised_draft = output[output.find('<result>')+8:output.find('</result>')]
                    
#                 #print('--> draft: ', draft)
#                 print('--> reflection: ', reflection)
#                 print('--> revised_draft: ', revised_draft)

#                 references += docs
#                 print('len(references): ', len(references))
#             else:
#                 print('No relevant document!')
#                 revised_draft = draft
#         else:
#             print('No reflection!')
#             revised_draft = draft
            
#         revision_number = state["revision_number"] if state.get("revision_number") is not None else 1

#         print('----> length of references: ', len(references))
        
#         return {
#             "revised_draft": revised_draft,            
#             "revision_number": revision_number,
#             "references": references
#         }
        
#     MAX_REVISIONS = 1
    
#     def reflect_node(state: ReflectionState, config):
#         print("###### reflect ######")
#         draft = state['draft']
        
#         if debugMode=="Debug":
#             st.info(f"draft에서 개선 사항을 도출합니다.")
    
#         reflection = []
#         search_queries = []
#         for attempt in range(10):
#             chat = get_chat()
#             if isKorean(draft):
#                 structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
#             else:
#                 structured_llm = chat.with_structured_output(Research, include_raw=True)
            
#             try:
#                 print('draft: ', draft)
#                 info = structured_llm.invoke(draft)
#                 print(f'attempt: {attempt}, info: {info}')
                    
#                 if not info['parsed'] == None:
#                     parsed_info = info['parsed']
#                     # print('reflection: ', parsed_info.reflection)                
#                     reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
#                     search_queries = parsed_info.search_queries

#                     if debugMode=="Debug":
#                         st.info(f"개선사항: {reflection}")
                    
#                     print('reflection: ', parsed_info.reflection)
#                     print('search_queries: ', search_queries)
            
#                     if isKorean(draft):
#                         translated_search = []
#                         for q in search_queries:
#                             chat = get_chat()
#                             if isKorean(q):
#                                 search = traslation(chat, q, "Korean", "English")
#                             else:
#                                 search = traslation(chat, q, "English", "Korean")
#                             translated_search.append(search)
                            
#                         print('translated_search: ', translated_search)
#                         search_queries += translated_search

#                     if debugMode=="Debug":
#                         st.info(f"검색어: {search_queries}")

#                     print('search_queries (mixed): ', search_queries)
#                     break
#             except Exception:
#                 print('---> parsing error for the draft')

#                 err_msg = traceback.format_exc()
#                 print('error message: ', err_msg)
#                 # raise Exception ("Not able to request to LLM")
            
#         revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
#         return {
#             "reflection": reflection,
#             "search_queries": search_queries,
#             "revision_number": revision_number + 1
#         }

#     def retrieve_docs(search_queries, config):
#         parallel_retrieval = config.get("configurable", {}).get("parallel_retrieval")
#         print('parallel_retrieval: ', parallel_retrieval)
        
#         docs = []        

#         for q in search_queries:      
#             if debugMode=="Debug":
#                 st.info(f"검색을 수행합니다. 검색어: {q}")

#             #docs = retrieve_documents_from_knowledge_base(q, top_k=numberOfDocs)
#             tavily_docs = retrieve_documents_from_tavily(q, top_k=numberOfDocs)
        
#             # grade
#             docs += grade_documents(q, tavily_docs) # grading
            
#         docs = check_duplication(docs) # check duplication
#         for i, doc in enumerate(docs):
#             print(f"#### {i}: {doc.page_content[:100]}")
            
#         return docs

#     def buildReflection():
#         workflow = StateGraph(ReflectionState)

#         # Add nodes
#         workflow.add_node("reflect_node", reflect_node)
#         workflow.add_node("revise_draft", revise_draft)

#         # Set entry point
#         workflow.set_entry_point("reflect_node")
        
#         workflow.add_conditional_edges(
#             "revise_draft", 
#             should_continue, 
#             {
#                 "end": END, 
#                 "continue": "reflect_node"
#             }
#         )

#         # Add edges
#         workflow.add_edge("reflect_node", "revise_draft")
        
#         return workflow.compile()
    
#     # initiate
#     global reference_docs, contentList
#     reference_docs = []
#     contentList =  []

#     # workflow
#     reflection_app = buildReflection()
                
#     final_doc = ""   
#     references = []

#     inputs = {
#         "draft": draft
#     }                    
#     app_config = {
#         "recursion_limit": 50,
#         "max_revisions": MAX_REVISIONS,
#         "parallel_retrieval": "disable"
#     }
#     output = reflection_app.invoke(inputs, config=app_config)
#     final_doc += output['revised_draft'] + '\n\n'

#     if 'references' in output:
#         references += output['references']

#     print('references: ', references)

#     markdown_references = ""
#     if references:
#         markdown_references = get_references(references)

#     return final_doc+markdown_references


