import traceback
import boto3
import os
import json
import re
import PyPDF2
import uuid
import base64
import csv
import info # user defined info such as models
import yfinance as yf
import utils
import rag_opensearch as rag
import search

from io import BytesIO
from PIL import Image
from pytz import timezone
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from langchain.docstore.document import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from typing import List
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

from multiprocessing import Process, Pipe
from urllib import parse
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = utils.CreateLogger("chat")

userId = "demo"
map_chain = dict() 

def initiate():
    global userId
    global memory_chain
    
    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")

    if userId in map_chain:  
            # print('memory exist. reuse it!')
            memory_chain = map_chain[userId]
    else: 
        # print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

initiate()

# load config
try:
    with open("/home/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        logger.info(f"config: {config}")

except Exception:
    print("use local configuration")
    logger.info(f"use local configuration")
    with open("application/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        logger.info(f"config: {config}")

bedrock_region = config["region"] if "region" in config else "us-west-2"

projectName = config["projectName"] if "projectName" in config else "langgraph-nova"

accountId = config["accountId"] if "accountId" in config else None
if accountId is None:
    raise Exception ("No accountId")

region = config["region"] if "region" in config else "us-west-2"
logger.info(f"region: {region}")

s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None
if s3_bucket is None:
    raise Exception ("No storage!")

s3_prefix = 'docs'
s3_image_prefix = 'images'

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

parsingModelArn = f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
embeddingModelArn = f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"

knowledge_base_name = projectName

numberOfDocs = 4
MSG_LENGTH = 100    
grade_state = "LLM" # LLM, PRIORITY_SEARCH, OTHERS

doc_prefix = s3_prefix+'/'
useEnhancedSearch = False

# RAG
index_name = projectName

LLM_embedding = json.loads(config["LLM_embedding"]) if "LLM_embedding" in config else None
if LLM_embedding is None:
    raise Exception ("No Embedding!")


enableParentDocumentRetrival = 'true'
enableHybridSearch = 'true'
selected_embedding = 0

model_name = "Nova Pro"
model_type = "nova"
multi_region = 'Enable'
contextual_embedding = "Disable"
debug_mode = "Enable"

models = info.get_model_info(model_name)
number_of_models = len(models)
selected_chat = 0

def update(modelName, debugMode, multiRegion, contextualEmbedding):    
    global model_name, debug_mode, multi_region, contextual_embedding     
    global selected_chat, selected_embedding, models, number_of_models
    
    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")
        
        selected_chat = 0
        selected_embedding = 0
        models = info.get_model_info(model_name)
        number_of_models = len(models)
        
    if debug_mode != debugMode:
        debug_mode = debugMode
        logger.info(f"debug_mode: {debug_mode}")

    if multi_region != multiRegion:
        multi_region = multiRegion
        logger.info(f"multi_region: {multi_region}")
        
        selected_chat = 0
        selected_embedding = 0

    if contextual_embedding != contextualEmbedding:
        contextual_embedding = contextualEmbedding
        logger.info(f"contextual_embedding: {contextual_embedding}")

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
    global selected_chat, model_type

    profile = models[selected_chat]
    # print('profile: ', profile)
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k
    logger.info(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
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
    
    if multi_region=='Enable':
        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    else:
        selected_chat = 0

    return chat

def get_parallel_processing_chat(models, selected):
    global model_type
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    maxOutputTokens = 4096
    logger.info(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
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
            
    logger.info(f"{i}: {text}, metadata:{doc.metadata}")

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
        logger.info(f"No secret found for weather api")

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
        logger.info(f"No secret found for lengsmith api")
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

# secret of code interpreter
code_interpreter_api_key = ""
try:
    get_code_interpreter_api_secret = secretsmanager.get_secret_value(
        SecretId=f"code-interpreter-{projectName}"
    )
    #print('get_code_interpreter_api_secret: ', get_code_interpreter_api_secret)
    secret = json.loads(get_code_interpreter_api_secret['SecretString'])
    #print('secret: ', secret)
    code_interpreter_api_key = secret['code_interpreter_api_key']
    code_interpreter_project = secret['project_name']
    code_interpreter_id = secret['code_interpreter_id']

    # logger.info(f"code_interpreter_id: {code_interpreter_id}")
except Exception as e:
    raise e

if code_interpreter_api_key:
    os.environ["RIZA_API_KEY"] = code_interpreter_api_key

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
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
        logger.info(f"error message: {err_msg}")

        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def upload_to_s3(file_bytes, file_name, contextual_embedding):
    """
    Upload a file to S3 and return the URL
    """
    try:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )

        # Generate a unique file name to avoid collisions
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #unique_id = str(uuid.uuid4())[:8]
        #s3_key = f"uploaded_images/{timestamp}_{unique_id}_{file_name}"
        s3_key = f"{s3_prefix}/{file_name}"

        content_type = utils.get_contents_type(file_name)       
        logger.info(f"content_type: {content_type}") 
        
        user_meta = {  # user-defined metadata
            "content_type": content_type,
            "model_name": model_name,
            "contextual_embedding": contextual_embedding,
            "multi_region": multi_region
        }
        
        response = s3_client.put_object(
            Bucket=s3_bucket, 
            Key=s3_key, 
            ContentType=content_type,
            Metadata = user_meta,
            Body=file_bytes            
        )
        logger.info(f"upload response: {response}")

        url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        return url
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        logger.info(f"{err_msg}")
        return None

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_parallel_processing_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        logger.info(f"---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        logger.info(f"--GRADE: DOCUMENT NOT RELEVANT---")
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
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, models, selected_chat))
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
    
    # from langchain_core.output_parsers import PydanticOutputParser  # not supported for Nova
    # parser = PydanticOutputParser(pydantic_object=GradeDocuments)
    # retrieval_grader = grade_prompt | chat | parser

    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_documents(question, documents):
    logger.info(f"###### grade_documents ######")
    
    logger.info(f"start grading...")
    logger.info(f"grade_state: {grade_state}")
    
    if grade_state == "LLM":
        filtered_docs = []
        if multi_region == 'Enable':  # parallel processing        
            filtered_docs = grade_documents_using_parallel_processing(question, documents)

        else:
            # Score each doc    
            llm = get_chat()
            retrieval_grader = get_retrieval_grader(llm)
            for i, doc in enumerate(documents):
                # print('doc: ', doc)
                print_doc(i, doc)
                
                score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                # print("score: ", score)
                
                grade = score.binary_score
                # print("grade: ", grade)
                # Document relevant
                if grade.lower() == "yes":
                    logger.info(f"---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(doc)
                # Document not relevant
                else:
                    logger.info(f"---GRADE: DOCUMENT NOT RELEVANT---")
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
    logger.info(f"length of relevant_docs: {len(docs)}")
    for doc in docs:            
        if doc.page_content in contentList:
            logger.info(f"duplicated")
            continue
        contentList.append(doc.page_content)
        updated_docs.append(doc)            
    length_updated_docs = len(updated_docs)     
    
    if length_original == length_updated_docs:
        logger.info(f"no duplication")
    else:
        logger.info(f"length of updated relevant_docs: {length_updated_docs}")
    
    return updated_docs

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
            logger.info(f"url: {url}")
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
        logger.info(f"excerpt(quotation removed): {excerpt}")

        name = name[name.rfind('/')+1:]
        
        if page:                
            reference += f"{i+1}. {page}page in [{name}]({url})), {excerpt[:30]}...\n"
        else:
            reference += f"{i+1}. [{name}]({url}), {excerpt[:30]}...\n"

    if reference: 
        reference = "\n\n#### 관련 문서\n"+reference

    return reference

def extract_thinking_tag(response, st):
    if response.find('<thinking>') != -1:
        status = response[response.find('<thinking>')+11:response.find('</thinking>')]
        logger.info(f"agent_thinking: {status}")
        
        if debug_mode=="Enable":
            st.info(status)

        if response.find('<thinking>') == 0:
            msg = response[response.find('</thinking>')+13:]
        else:
            msg = response[:response.find('<thinking>')]
        logger.info(f"msg: {msg}")
    else:
        msg = response

    return msg

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    logger.info(f"lins: {len(lines)}")
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    logger.info(f"columns: {columns}")
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    logger.info(f"docs[0]: {docs[0]}")

    return docs

def get_summary(docs):    
    llm = get_chat()

    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        logger.info(f"result of summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    return summary

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    contents = ""
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt' or file_type == 'md':
        contents = doc.get()['Body'].read().decode('utf-8')
        
    logger.info(f"contents: {contents}")
    new_contents = str(contents).replace("\n"," ") 
    logger.info(f"length: {len(new_contents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 
    texts = text_splitter.split_text(new_contents) 
    if texts:
        logger.info(f"texts[0]: {texts[0]}")
    
    return texts

def summary_of_code(code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다." 
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    llm = get_chat()

    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        logger.info(f"esult of code summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def summary_image(img_base64, instruction):      
    llm = get_chat()

    if instruction:
        logger.info(f"instruction: {instruction}")
        query = f"{instruction}. <result> tag를 붙여주세요."
    else:
        query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. markdown 포맷으로 답변을 작성합니다."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = llm.invoke(messages)
            
            extracted_text = result.content
            # print('summary from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            # raise Exception ("Not able to request to LLM")
        
    return extracted_text

def extract_text(img_base64):    
    multimodal = get_chat()
    query = "텍스트를 추출해서 markdown 포맷으로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = multimodal.invoke(messages)
            
            extracted_text = result.content
            # print('result of text extraction from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            # raise Exception ("Not able to request to LLM")
    
    logger.info(f"extracted_text: {extracted_text}")
    if len(extracted_text)<10:
        extracted_text = "텍스트를 추출하지 못하였습니다."    

    return extracted_text

fileId = uuid.uuid4().hex
# print('fileId: ', fileId)
def get_summary_of_uploaded_file(file_name, st):
    file_type = file_name[file_name.rfind('.')+1:len(file_name)]            
    logger.info(f"file_type: {file_type}")
    
    if file_type == 'csv':
        docs = load_csv_document(file_name)
        contexts = []
        for doc in docs:
            contexts.append(doc.page_content)
        logger.info(f"contexts: {contexts}")
    
        msg = get_summary(contexts)

    elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
        texts = load_document(file_type, file_name)

        docs = []
        for i in range(len(texts)):
            docs.append(
                Document(
                    page_content=texts[i],
                    metadata={
                        'name': file_name,
                        # 'page':i+1,
                        'url': path+doc_prefix+parse.quote(file_name)
                    }
                )
            )
        logger.info(f"docs[0]: {docs[0]}")
        logger.info(f"docs size: {len(docs)}")

        contexts = []
        for doc in docs:
            contexts.append(doc.page_content)
        logger.info(f"contexts: {contexts}")

        msg = get_summary(contexts)
        
    elif file_type == 'py' or file_type == 'js':
        s3r = boto3.resource("s3")
        doc = s3r.Object(s3_bucket, s3_prefix+'/'+file_name)
        
        contents = doc.get()['Body'].read().decode('utf-8')
        
        #contents = load_code(file_type, object)                
                        
        msg = summary_of_code(contents, file_type)                  
        
    elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
        logger.info(f"multimodal: {file_name}")
        
        s3_client = boto3.client('s3') 
            
        if debug_mode=="Enable":
            status = "이미지를 가져옵니다."
            logger.info(f"status: {status}")
            st.info(status)
            
        image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+file_name)
        # print('image_obj: ', image_obj)
        
        image_content = image_obj['Body'].read()
        img = Image.open(BytesIO(image_content))
        
        width, height = img.size 
        logger.info(f"width: {width}, height: {height}, size: {width*height}")
        
        isResized = False
        while(width*height > 5242880):                    
            width = int(width/2)
            height = int(height/2)
            isResized = True
            logger.info(f"pwidth: {width}, height: {height}, size: {width*height}")
        
        if isResized:
            img = img.resize((width, height))
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
               
        # extract text from the image
        if debug_mode=="Enable":
            status = "이미지에서 텍스트를 추출합니다."
            logger.info(f"status: {status}")
            st.info(status)
        
        text = extract_text(img_base64)
        # print('extracted text: ', text)

        if text.find('<result>') != -1:
            extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
            # print('extracted_text: ', extracted_text)
        else:
            extracted_text = text

        if debug_mode=="Enable":
            status = f"### 추출된 텍스트\n\n{extracted_text}"
            logger.info(f"status: {status}")
            st.info(status)
    
        if debug_mode=="Enable":
            status = "이미지의 내용을 분석합니다."
            logger.info(f"status: {status}")
            st.info(status)

        image_summary = summary_image(img_base64, "")

        if image_summary.find('<result>') != -1:
            image_summary = image_summary[image_summary.find('<result>')+8:image_summary.find('</result>')] # remove <result> tag        
        logger.info(f"image summary: {image_summary}")
            
        if len(extracted_text) > 10:
            contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
        else:
            contents = f"## 이미지 분석\n\n{image_summary}"
        logger.info(f"image contents: {contents}")

        msg = contents

    global fileId
    fileId = uuid.uuid4().hex
    # print('fileId: ', fileId)

    return msg

def revise_question(query, st):    
    logger.info(f"###### revise_question ######")

    llm = get_chat()
    st.info("히스토리를 이용해 질문을 변경합니다.")
        
    if isKorean(query)==True :      
        human = (
            "이전 대화를 참조하여, 다음의 <question>의 뜻을 명확히 하는 새로운 질문을 한국어로 생성하세요." 
            "새로운 질문은 원래 질문의 중요한 단어를 반드시 포함합니다." 
            "결과는 <result> tag를 붙여주세요."
        
            "<question>"
            "{question}"
            "</question>"
        )
        
    else: 
        human = (
            "Rephrase the follow up <question> to be a standalone question." 
            "Put it in <result> tags."

            "<question>"
            "{question}"
            "</question>"
        )
            
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)]
    )
    # print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    logger.info(f"history: {history}")

    if not len(history):        
        print('no history')
        st.info("이전 히스트로가 없어서 질문을 그대로 전달합니다.")
        return query
                
    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "history": history,
                "question": query,
            }
        )
        generated_question = result.content
        
        revised_question = generated_question[generated_question.find('<result>')+8:len(generated_question)-9] # remove <result> tag                   
        # print('revised_question: ', revised_question)

        st.info(f"수정된 질문: {revised_question}")
        
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")        
        raise Exception ("Not able to request to LLM")
            
    return revised_question    

####################### LangChain #######################
# General Conversation
#########################################################
def general_conversation(query):
    llm = get_chat()

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

    chain = prompt | llm | StrOutputParser()
    try: 
        stream = chain.stream(
            {
                "history": history,
                "input": query,
            }
        )  
            
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")        
        raise Exception ("Not able to request to LLM: "+err_msg)
        
    return stream

####################### LangChain #######################
# Basic RAG (OpenSearch)
#########################################################
def get_embedding():
    global selected_embedding
    embedding_profile = LLM_embedding[selected_embedding]
    bedrock_region =  embedding_profile['bedrock_region']
    model_id = embedding_profile['model_id']
    logger.info(f"selected_embedding: {selected_embedding}, bedrock_region: {bedrock_region}, model_id: {model_id}")
    
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
    
    bedrock_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = model_id
    )  
    
    if multi_region=='Enable':
        selected_embedding = selected_embedding + 1
        if selected_embedding == len(LLM_embedding):
            selected_embedding = 0
    else:
        selected_embedding = 0

    return bedrock_embedding

def get_rag_prompt(text):
    # print("###### get_rag_prompt ######")
    llm = get_chat()
    # print('model_type: ', model_type)
    
    if model_type == "nova":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Provide a concise answer to the question at the end using reference texts." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
            )    
    
        human = (
            "Question: {question}"

            "Reference texts: "
            "{context}"
        ) 
        
    elif model_type == "claude":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Here is pieces of context, contained in <context> tags." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
                "Put it in <result> tags."
            )    

        human = (
            "<question>"
            "{question}"
            "</question>"

            "<context>"
            "{context}"
            "</context>"
        )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    rag_chain = prompt | llm

    return rag_chain

def get_answer_using_opensearch(text, st):
    # retrieve
    if debug_mode == "Enable":
        st.info(f"RAG 검색을 수행합니다. 검색어: {text}")        
    relevant_docs = rag.retrieve_documents_from_opensearch(text, top_k=4)
        
    # grade   
    if debug_mode == "Enable":
        st.info(f"가져온 {len(relevant_docs)}개의 문서를 평가하고 있습니다.")     
    filtered_docs = grade_documents(text, relevant_docs) # grading    
    filtered_docs = check_duplication(filtered_docs) # check duplication

    if debug_mode == "Enable":
        st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")
    
    # generate
    if debug_mode == "Enable":
        st.info(f"결과를 생성중입니다.")
    relevant_context = ""
    for document in filtered_docs:
        relevant_context = relevant_context + document.page_content + "\n\n"        
    # print('relevant_context: ', relevant_context)

    rag_chain = get_rag_prompt(text)                       
    msg = ""    
    try: 
        result = rag_chain.invoke(
            {
                "question": text,
                "context": relevant_context                
            }
        )
        logger.info(f"result: {result}")

        msg = result.content        
        if msg.find('<result>')!=-1:
            msg = msg[msg.find('<result>')+8:msg.find('</result>')]
        
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    reference = ""
    if filtered_docs:
        reference = get_references(filtered_docs)
    
    return msg+reference, filtered_docs

####################### LangGraph #######################
# Corrective RAG
#########################################################
isKorPrompt = False

def get_rewrite():
    class RewriteQuestion(BaseModel):
        """rewrited question that is well optimized for retrieval."""

        question: str = Field(description="The new question is optimized to represent semantic intent and meaning of the user")
    
    llm = get_chat()
    structured_llm_rewriter = llm.with_structured_output(RewriteQuestion)
    
    logger.info(f"isKorPrompt: {isKorPrompt}")
    
    if isKorPrompt:
        system = """당신은 질문 re-writer입니다. 사용자의 의도와 의미을 잘 표현할 수 있도록 질문을 한국어로 re-write하세요."""
    else:
        system = (
            "You a question re-writer that converts an input question to a better version that is optimized"
            "for web search. Look at the input and try to reason about the underlying semantic intent / meaning."
        )
        
    logger.info(f"system: {system}")
        
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}"),
        ]
    )
    question_rewriter = re_write_prompt | structured_llm_rewriter
    return question_rewriter

def get_hallucination_grader():    
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )
        
    system = (
        "You are a grader assessing whether an LLM generation is grounded in supported by a set of retrieved facts."
        "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in supported by the set of facts."
    )    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
        
    llm = get_chat()
    structured_llm_grade_hallucination = llm.with_structured_output(GradeHallucinations)
        
    hallucination_grader = hallucination_prompt | structured_llm_grade_hallucination
    return hallucination_grader

def run_corrective_rag(query, st):
    class State(TypedDict):
        question : str
        generation : str
        web_search : str
        documents : List[str]

    def retrieve_node(state: State):
        logger.info(f"p###### retrieve ######")
        question = state["question"]

        if debug_mode=="Enable":
            st.info(f"RAG 검색을 수행합니다. 검색어: {question}")        
        docs = rag.retrieve_documents_from_opensearch(question, top_k=4)
        
        return {"documents": docs, "question": question}

    def grade_documents_node(state: State, config):
        logger.info(f"###### grade_documents ######")
        question = state["question"]
        documents = state["documents"]
        
        if debug_mode=="Enable":
            st.info(f"가져온 {len(documents)}개의 문서를 평가하고 있습니다.")    
        
        # Score each doc
        filtered_docs = []
        logger.info(f"start grading...")
        logger.info(f"grade_state: ", grade_state)
        
        web_search = "No"
        
        if grade_state == "LLM":
            if multi_region == 'Enable':  # parallel processing            
                filtered_docs = grade_documents_using_parallel_processing(question, documents)
                
                if len(documents) != len(filtered_docs):
                    web_search = "Yes"

            else:    
                llm = get_chat()
                retrieval_grader = get_retrieval_grader(llm)
                for doc in documents:
                    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                    grade = score.binary_score
                    # Document relevant
                    if grade.lower() == "yes":
                        logger.info(f"---GRADE: DOCUMENT RELEVANT---")
                        filtered_docs.append(doc)
                    # Document not relevant
                    else:
                        logger.info(f"---GRADE: DOCUMENT NOT RELEVANT---")
                        # We do not include the document in filtered_docs
                        # We set a flag to indicate that we want to run web search
                        web_search = "Yes"
                        continue
            logger.info(f"len(documents): {len(filtered_docs)}")
            logger.info(f"web_search: {web_search}")
            
        # elif grade_state == "PRIORITY_SEARCH":
        #     filtered_docs = priority_search(question, documents, minDocSimilarity)
        else:  # OTHERS
            filtered_docs = documents

        filtered_docs = check_duplication(filtered_docs) # check duplication

        if debug_mode == "Enable":
            st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")
        
        global reference_docs
        reference_docs += filtered_docs
        
        return {"question": question, "documents": filtered_docs, "web_search": web_search}

    def decide_to_generate(state: State):
        logger.info(f"###### decide_to_generate ######")
        web_search = state["web_search"]
        
        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            logger.info(f"---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "rewrite"
        else:
            # We have relevant documents, so generate answer
            logger.info(f"---DECISION: GENERATE---")
            return "generate"

    def generate_node(state: State, config):
        logger.info(f"###### generate ######")
        question = state["question"]
        documents = state["documents"]

        if debug_mode=="Enable":
            st.info(f"답변을 생성하고 있습니다.")       
        
        # RAG generation
        rag_chain = get_rag_prompt(question)

        relevant_context = ""
        for document in documents:
            relevant_context = relevant_context + document.page_content + "\n\n"        
        # print('relevant_context: ', relevant_context)
        
        result = rag_chain.invoke(
            {
                "question": question, 
                "context": relevant_context
            }
        )
        logger.info(f"result: {result}")

        output = result.content
        if output.find('<result>')!=-1:
            output = output[output.find('<result>')+8:output.find('</result>')]

        if len(documents):
            global reference_docs
            reference_docs += documents
        
        return {"generation": output}

    def rewrite_node(state: State, config):
        logger.info(f"###### rewrite ######")
        question = state["question"]
        documents = state["documents"]

        if debug_mode=="Enable":
            st.info(f"질문을 새로 생성하고 있습니다.")       
        
        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        logger.info(f"better_question: {better_question.question}")

        return {"question": better_question.question, "documents": documents}

    def web_search_node(state: State, config):
        logger.info(f"###### web_search ######")
        question = state["question"]
        documents = state["documents"]

        if debug_mode=="Enable":
            st.info(f"인터넷을 검색합니다. 검색어: {question}")
        
        docs = search.retrieve_documents_from_tavily(question, top_k=3)
        # print('docs: ', docs)

        if debug_mode == "Enable":
            st.info(f"{len(docs)}개의 문서가 검색되었습니다.")

        for doc in docs:
            documents.append(doc)
        # print('documents: ', documents)
            
        return {"question": question, "documents": documents}

    def buildCorrectiveRAG():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve_node)  
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("websearch", web_search_node)

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "rewrite": "rewrite",
                "generate": "generate",
            },
        )
        workflow.add_edge("rewrite", "websearch")
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    app = buildCorrectiveRAG()

    global contentList, reference_docs
    contentList = []
    reference_docs = []
    
    global isKorPrompt
    isKorPrompt = isKorean(query)
            
    inputs = {"question": query}
    config = {
        "recursion_limit": 50
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
            # print("value: ", value)
            
    #print('value: ', value)

    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)

    return value["generation"] + reference, reference_docs

####################### LangGraph #######################
# Self RAG
#########################################################
MAX_RETRIES = 2 # total 3

def get_answer_grader():
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )
        
    llm = get_chat()
    structured_llm_grade_answer = llm.with_structured_output(GradeAnswer)
        
    system = (
        "You are a grader assessing whether an answer addresses / resolves a question."
        "Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_grade_answer
    return answer_grader

class GraphConfig(TypedDict):
    max_retries: int    
    max_count: int

def run_self_rag(query, st):
    class State(TypedDict):
        question : str
        generation : str
        retries: int  # number of generation 
        count: int # number of retrieval
        documents : List[str]
    
    def retrieve_node(state: State, config):
        logger.info(f"state: {state}")
        logger.info(f"###### retrieve ######")
        question = state["question"]

        if debug_mode=="Enable":
            st.info(f"RAG 검색을 수행합니다. 검색어: {question}")        
        docs = rag.retrieve_documents_from_opensearch(question, top_k=4)
        
        return {"documents": docs, "question": question}
    
    def generate_node(state: State, config):
        logger.info(f"###### generate ######")
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1

        if debug_mode=="Enable":
            st.info(f"답변을 생성합니다.")
        
        # RAG generation
        rag_chain = get_rag_prompt(question)

        relevant_context = ""
        for document in documents:
            relevant_context = relevant_context + document.page_content + "\n\n"        
        # print('relevant_context: ', relevant_context)
            
        result = rag_chain.invoke(
            {
                "question": question,
                "context": relevant_context                
            }
        )
        logger.info(f"result: {result}")

        output = result.content
        if output.find('<result>')!=-1:
            output = output[output.find('<result>')+8:output.find('</result>')]
        
        return {"generation": output, "retries": retries + 1}
            
    def grade_documents_node(state: State, config):
        print("###### grade_documents ######")
        question = state["question"]
        documents = state["documents"]
        count = state["count"] if state.get("count") is not None else -1

        if debug_mode=="Enable":
            st.info(f"가져온 {len(documents)}개의 문서를 평가하고 있습니다.")    
        
        logger.info(f"start grading...")
        logger.info(f"grade_state: {grade_state}")
        
        if grade_state == "LLM":
            if multi_region == 'Enable':  # parallel processing            
                filtered_docs = grade_documents_using_parallel_processing(question, documents)

            else:    
                # Score each doc
                filtered_docs = []
                llm = get_chat()
                retrieval_grader = get_retrieval_grader(llm)
                for doc in documents:
                    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                    grade = score.binary_score
                    # Document relevant
                    if grade.lower() == "yes":
                        logger.info(f"---GRADE: DOCUMENT RELEVANT---")
                        filtered_docs.append(doc)
                    # Document not relevant
                    else:
                        logger.info(f"---GRADE: DOCUMENT NOT RELEVANT---")
                        # We do not include the document in filtered_docs
                        # We set a flag to indicate that we want to run web search
                        continue
            logger.info(f"len(docments): {len(filtered_docs)}")

        # elif grade_state == "PRIORITY_SEARCH":
        #     filtered_docs = priority_search(question, documents, minDocSimilarity)
        else:  # OTHERS
            filtered_docs = documents

        filtered_docs = check_duplication(filtered_docs) # check duplication

        if debug_mode == "Enable":
            st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")
        
        global reference_docs
        reference_docs += filtered_docs
        
        return {"question": question, "documents": filtered_docs, "count": count + 1}

    def decide_to_generate(state: State, config):
        logger.info(f"###### decide_to_generate ######")
        filtered_documents = state["documents"]
        
        count = state["count"] if state.get("count") is not None else -1
        max_count = config.get("configurable", {}).get("max_count", MAX_RETRIES)
        logger.info(f"count: {count}")
        
        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            logger.info(f"---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "no document" if count < max_count else "not available"
        else:
            # We have relevant documents, so generate answer
            logger.info(f"---DECISION: GENERATE---")
            return "document"

    def rewrite_node(state: State, config):
        logger.info(f"###### rewrite ######")
        question = state["question"]
        documents = state["documents"]

        if debug_mode=="Enable":
            st.info(f"질문을 새로 생성합니다.")       
        
        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        logger.info(f"better_question: {better_question.question}")

        return {"question": better_question.question, "documents": documents}

    def grade_generation(state: State, config):
        logger.info(f"###### grade_generation ######")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        global reference_docs

        retries = state["retries"] if state.get("retries") is not None else -1
        max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

        logger.info(f"len(documents): {len(documents)}")

        if len(documents):
            # Check Hallucination
            if debug_mode=="Enable":
                st.info(f"환각(hallucination)인지 검토합니다.")

            hallucination_grader = get_hallucination_grader()        

            hallucination_grade = "no"
            for attempt in range(3):   
                logger.info(f"attempt: {attempt}")
                try:
                    score = hallucination_grader.invoke(
                        {"documents": documents, "generation": generation}
                    )
                    hallucination_grade = score.binary_score
                    break
                except Exception:
                    err_msg = traceback.format_exc()
                    logger.info(f"error message: {err_msg}")       
            
            logger.info(f"hallucination_grade: {hallucination_grade}")
            logger.info(f"retries: {retries}")
        else:
            hallucination_grade = "yes" # not hallucination
            if debug_mode=="Enable":
                st.info(f"검색된 문서가 없어서 환격(hallucination)은 테스트하지 않습니다.")

        answer_grader = get_answer_grader()
        if hallucination_grade == "yes":
            if debug_mode=="Enable" and len(documents):
                st.info(f"환각이 아닙니다.")

            logger.info(f"---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

            # Check appropriate answer
            if debug_mode=="Enable":
                st.info(f"적절한 답변인지 검토합니다.")

            logger.info(f"---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            answer_grade = score.binary_score        
            # print("answer_grade: ", answer_grade)

            if answer_grade == "yes":
                logger.info(f"---DECISION: GENERATION ADDRESSES QUESTION---")
                if debug_mode=="Enable":
                    st.info(f"적절한 답변입니다.")
                return "useful" 
            else:
                logger.info(f"---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                if debug_mode=="Enable":
                    st.info(f"적절하지 않은 답변입니다.")
                
                
                reference_docs = []
                return "not useful" if retries < max_retries else "not available"
        else:
            logger.info(f"---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            if debug_mode=="Enable":
                st.info(f"환각(halucination)입니다.")
            reference_docs = []
            return "not supported" if retries < max_retries else "not available"
        
    def build():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve_node)  
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("rewrite", rewrite_node)

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "no document": "rewrite",
                "document": "generate",
                "not available": "generate",
            },
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "rewrite",
                "not available": END,
            },
        )

        return workflow.compile()

    app = build()    
    
    global isKorPrompt
    isKorPrompt = isKorean(query)

    global contentList, reference_docs
    contentList = []
    reference_docs = []
        
    inputs = {"question": query}
    config = {
        "recursion_limit": 50
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
            # print("value: ", value)
            
    #print('value: ', value)

    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)
        
    return value["generation"] + reference, reference_docs

####################### LangGraph #######################
# Self Corrective RAG
#########################################################
def run_self_corrective_rag(query, st):
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        question: str
        documents: list[Document]
        candidate_answer: str
        retries: int
        web_fallback: bool

    def retrieve_node(state: State, config):
        logger.info(f"###### retrieve ######")
        question = state["question"]
        
        if debug_mode=="Enable":
            st.info(f"RAG 검색을 수행합니다. 검색어: {question}")
        
        docs = rag.retrieve_documents_from_opensearch(question, top_k=4)

        if debug_mode == "Enable":
            st.info(f"{len(docs)}개의 문서가 선택되었습니다.")
        
        return {"documents": docs, "question": question, "web_fallback": True}

    def generate_node(state: State, config):
        logger.info(f"###### generate ######")
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1

        if debug_mode=="Enable":
            st.info(f"답변을 생성합니다.")
                
        # RAG generation
        rag_chain = get_rag_prompt(question)

        relevant_context = ""
        for document in documents:
            relevant_context = relevant_context + document.page_content + "\n\n"        
        # print('relevant_context: ', relevant_context)
            
        result = rag_chain.invoke(
            {
                "question": question,
                "context": relevant_context                
            }
        )
        logger.info(f"result: {result}")

        output = result.content
        if output.find('<result>')!=-1:
            output = output[output.find('<result>')+8:output.find('</result>')]
        
        global reference_docs
        reference_docs += documents
        
        return {"retries": retries + 1, "candidate_answer": output}

    def rewrite_node(state: State, config):
        logger.info(f"###### rewrite ######")
        question = state["question"]
        
        if debug_mode=="Enable":
            st.info(f"질문을 새로 생성하고 있습니다.")      

        # Prompt
        question_rewriter = get_rewrite()
        
        better_question = question_rewriter.invoke({"question": question})
        logger.info(f"better_question: {better_question.question}")

        return {"question": better_question.question, "documents": []}
    
    def grade_generation(state: State, config):
        print("###### grade_generation ######")
        logger.info(f"###### grade_generation ######")
        question = state["question"]
        documents = state["documents"]
        generation = state["candidate_answer"]
        web_fallback = state["web_fallback"]

        global reference_docs
                
        retries = state["retries"] if state.get("retries") is not None else -1
        max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

        if not web_fallback:
            return "finalize_response"        
        
        logger.info(f"len(documents):: {len(documents)}")
                
        if len(documents):
            # Check hallucination
            if debug_mode=="Enable":
                st.info(f"환각(hallucination)인지 검토합니다.")

            logger.info(f"---Hallucination?---")    
            hallucination_grader = get_hallucination_grader()
            hallucination_grade = "no"

            for attempt in range(3):   
                logger.info(f"attempt: {attempt}")
                
                try:        
                    score = hallucination_grader.invoke(
                        {"documents": documents, "generation": generation}
                    )
                    hallucination_grade = score.binary_score
                    break
                except Exception:
                    err_msg = traceback.format_exc()
                    logger.info(f"error message: {err_msg}")
        else:
            hallucination_grade = "yes" # not hallucination
            if debug_mode=="Enable":
                st.info(f"검색된 문서가 없어서 환격(hallucination)은 테스트하지 않습니다.")            

        if hallucination_grade == "no":
            logger.info(f"---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS (Hallucination), RE-TRY---")
            if debug_mode=="Enable":
                st.info(f"환각(halucination)입니다.")                        
            reference_docs = []
            return "generate" if retries < max_retries else "websearch"

        if debug_mode=="Enable" and len(documents):
            st.info(f"환각이 아닙니다.")
        
        logger.info(f"---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        logger.info(f"---GRADE GENERATION vs QUESTION---")

        # Check appropriate answer
        if debug_mode=="Enable":
            st.info(f"적절한 답변인지 검토합니다.")
        
        answer_grader = get_answer_grader()    

        for attempt in range(3):   
            logger.info(f"attempt: {attempt}")
            try: 
                score = answer_grader.invoke({"question": question, "generation": generation})
                answer_grade = score.binary_score     
                logger.info(f"answer_grade: {answer_grade}")
            
                break
            except Exception:
                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}")   
            
        if answer_grade == "yes":
            logger.info(f"---DECISION: GENERATION ADDRESSES QUESTION---")
            if debug_mode=="Enable":
                st.info(f"적절한 답변입니다.")
            return "finalize_response"
        else:
            logger.info(f"---DECISION: GENERATION DOES NOT ADDRESS QUESTION (Not Answer)---")
            if debug_mode=="Enable":
                st.info(f"적절하지 않은 답변입니다.")            
            reference_docs = []
            return "rewrite" if retries < max_retries else "websearch"

    def web_search_node(state: State, config):
        logger.info(f"###### web_search ######")
        question = state["question"]
        # documents = state["documents"]
        documents = [] # initiate 
        
        if debug_mode=="Enable":
            st.info(f"인터넷을 검색합니다. 검색어: {question}")
        
        docs = search.retrieve_documents_from_tavily(question, top_k=3)
        if debug_mode == "Enable":
            st.info(f"{len(docs)}개의 문서가 검색되었습니다.")

        for doc in docs:
            documents.append(doc)
        # print('documents: ', documents)
            
        return {"question": question, "documents": documents}

    def finalize_response_node(state: State):
        logger.info(f"###### finalize_response ######")
        return {"messages": [AIMessage(content=state["candidate_answer"])]}
        
    def buildSelCorrectivefRAG():
        workflow = StateGraph(State)
            
        # Define the nodes
        workflow.add_node("retrieve", retrieve_node)  
        workflow.add_node("generate", generate_node) 
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("websearch", web_search_node)
        workflow.add_node("finalize_response", finalize_response_node)

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("finalize_response", END)

        workflow.add_conditional_edges(
            "generate",
            grade_generation,
            {
                "generate": "generate",
                "websearch": "websearch",
                "rewrite": "rewrite",
                "finalize_response": "finalize_response",
            },
        )

        # Compile
        return workflow.compile()

    app = buildSelCorrectivefRAG()

    global isKorPrompt
    isKorPrompt = isKorean(query)

    global contentList, reference_docs
    contentList = []
    reference_docs = []
    
    inputs = {"question": query}
    config = {
        "recursion_limit": 50
    }
    
    for output in app.stream(inputs, config):   
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
            #print("value: ", value)
            
    #print('value: ', value)
    #print('content: ', value["messages"][-1].content)

    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)
        
    return value["messages"][-1].content + reference, reference_docs

####################### LangChain #######################
# Translation (English)
#########################################################
def translate_text(text, model_name):
    global llmMode
    llmMode = model_name

    llm = get_chat()

    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        logger.info(f"translated text: {msg}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")

    if msg.find('<result>') != -1:
        msg = msg[msg.find('<result>')+8:msg.find('</result>')] # remove <result> tag
    if msg.find('<article>') != -1:
        msg = msg[msg.find('<article>')+9:msg.find('</article>')] # remove <article> tag

    return msg

####################### LangChain #######################
# Image Summarization
#########################################################
def get_image_summarization(object_name, prompt, st):
    # load image
    s3_client = boto3.client(
        service_name='s3',
        region_name=bedrock_region
    )

    if debug_mode=="Enable":
        status = "이미지를 가져옵니다."
        logger.info(f"status: {status}")
        st.info(status)
                
    image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+object_name)
    # print('image_obj: ', image_obj)
    
    image_content = image_obj['Body'].read()
    img = Image.open(BytesIO(image_content))
    
    width, height = img.size 
    logger.info(f"width: {width}, height: {height}, size: {width*height}")
    
    isResized = False
    while(width*height > 5242880):                    
        width = int(width/2)
        height = int(height/2)
        isResized = True
        logger.info(f"width: {width}, height: {height}, size: {width*height}")
    
    if isResized:
        img = img.resize((width, height))
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # extract text from the image
    if debug_mode=="Enable":
        status = "이미지에서 텍스트를 추출합니다."
        logger.info(f"status: {status}")
        st.info(status)

    text = extract_text(img_base64)
    logger.info(f"extracted text: {text}")

    if text.find('<result>') != -1:
        extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
        # print('extracted_text: ', extracted_text)
    else:
        extracted_text = text
    
    if debug_mode=="Enable":
        status = f"### 추출된 텍스트\n\n{extracted_text}"
        logger.info(f"status: {status}")
        st.info(status)
    
    if debug_mode=="Enable":
        status = "이미지의 내용을 분석합니다."
        logger.info(f"status: {status}")
        st.info(status)

    image_summary = summary_image(img_base64, prompt)
    logger.info(f"image summary: {image_summary}")
        
    if len(extracted_text) > 10:
        contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
    else:
        contents = f"## 이미지 분석\n\n{image_summary}"
    logger.info(f"image contents: {contents}")

    return contents

