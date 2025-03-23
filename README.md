# Agentic RAG 구현하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Fagentic-rag&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false")](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue">
</p>

여기에서는 RAG의 성능 향상 기법인 Agentic RAG, Corrective RAG, Self RAG를 구현하는 방법을 설명합니다. 또한 RAG의 데이터 수집에 필요한 PDF의 header/footer의 처리, 이미지의 추출 및 분석과 함께 contextual retrieval을 활용하는 방법을 설명합니다. 이를 통해서 생성형 AI 애플리케이션을 위한  데이터를 효과적으로 수집하여 활용할 수 있습니다. 여기서는 오픈소스 LLM Framework인 [LangGraph](https://langchain-ai.github.io/langgraph/)을 이용하고, 구현된 workflow들은 [Streamlit](https://streamlit.io/)을 이용해 개발 및 테스트를 수행할 수 있습니다. [AWS CDK](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-construct-library.html)를 이용하고 한번에 배포할 수 있고, [CloudFront](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Introduction.html) - ALB 구조를 이용해 HTTPS로 안전하게 접속할 수 있습니다. 

## System Architecture 

전체적인 architecture는 아래와 같습니다. Streamlit이 설치된 EC2는 private subnet에 있고, CloudFront-ALB를 이용해 외부와 연결됩니다. RAG는 OpenSearch를 활용하고 있습니다. 인터넷 검색은 tavily를 사용하고 날씨 API를 추가로 활용합니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/3353ade1-db6e-4d30-baa5-be78d9820418" />

여기에서는 Lambda-Document를 이용해 입력된 문서를 parsing하여 OpenSearch에 push합니다. 이를 위한 event driven 방식의 데이터 처리 방식은 아래와 같습니다.

![image](https://github.com/user-attachments/assets/f89ac20b-91e3-490c-a34f-703c2957b022)


## 상세 구현

Agentic workflow (tool use)는 아래와 같이 구현할 수 있습니다. 상세한 내용은 [chat.py](./application/chat.py)을 참조합니다.

### Basic Chat

일반적인 대화는 아래와 같이 stream으로 결과를 얻을 수 있습니다. 여기에서는 LangChain의 ChatBedrock을 이용합니다. Model ID로 사용할 모델을 지정합니다. 아래 예제에서는 Nova Pro의 모델명인 "us.amazon.nova-pro-v1:0"을 활용하고 있습니다. Nova 모델는 동급 모델대비 빠르고, 높은 가성비와 함께 훌륭한 멀티모달 성능을 가지고 있습니다. 만약 Claude Sonnet 3.5을 사용한 다면 "anthropic.claude-3-5-sonnet-20240620-v1:0"을 입력합니다.
 
```python
modelId = "us.amazon.nova-pro-v1:0"
bedrock_region = "us-west-2"
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
    "stop_sequences": ["\n\n<thinking>", "\n<thinking>", " <thinking>"]
}

chat = ChatBedrock(  
    model_id=modelId,
    client=boto3_bedrock, 
    model_kwargs=parameters,
    region_name=bedrock_region
)

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
stream = chain.stream(
    {
        "history": history,
        "input": query,
    }
)  
print('stream: ', stream)
```

### Basic RAG

여기에서는 RAG 구현을 위하여 OpenSearch를 이용합니다. 

LangChain의 [OpenSearchVectorSearch](https://sj-langchain.readthedocs.io/en/latest/vectorstores/langchain.vectorstores.opensearch_vector_search.OpenSearchVectorSearch.html)을 이용하여 관련된 문서를 가져옵니다.

```python
vectorstore_opensearch = OpenSearchVectorSearch(
    index_name = index_name,
    is_aoss = False,
    ef_search = 1024,
    m=48,
    embedding_function = bedrock_embedding,
    opensearch_url=opensearch_url,
    http_auth=(opensearch_account, opensearch_passwd)
)
relevant_documents = vectorstore_opensearch.similarity_search_with_score(
    query = query,
    k = top_k
)
for i, document in enumerate(relevant_documents):
    name = document[0].metadata['name']
    url = document[0].metadata['url']
    content = document[0].page_content
```

얻어온 문서가 적절한지를 판단하기 위하여 아래와 같이 prompt를 이용해 관련도를 평가하고 [structured output](https://github.com/kyopark2014/langgraph-agent/blob/main/structured-output.md)을 이용해 관련도를 평가합니다.

```python
system = (
    "You are a grader assessing relevance of a retrieved document to a user question."
    "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
    
structured_llm_grader = chat.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader

filtered_docs = []
for i, doc in enumerate(documents):
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                
    grade = score.binary_score
    if grade.lower() == "yes":
        print("---GRADE: DOCUMENT RELEVANT---")
        filtered_docs.append(doc)
    else:
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        continue
```

이후 아래와 같이 RAG를 활용하여 원하는 응답을 얻습니다.

```python
system = (
  "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
  "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
  "모르는 질문을 받으면 솔직히 모른다고 말합니다."
  "답변의 이유를 풀어서 명확하게 설명합니다."
)
human = (
    "Question: {input}"

    "Reference texts: "
    "{context}"
)    
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
chain = prompt | chat
stream = chain.invoke(
    {
        "context": context,
        "input": revised_question,
    }
)
print(stream.content)    
```

### RAG의 성능 향상 방법

#### Hiearchical Chunking (Parent-Child Chunking)

문서를 크기에 따라 parent chunk와 child chunk로 나누어서 child chunk를 찾은 후에 LLM의 context에는 parent chunk를 사용하면, 검색의 정확도는 높이고 충분한 문서를 context로 활용할 수 있습니다. 아래에서는 parent doc을 생성후에 다시 child doc을 생성합니다. child doc은 metadata에 parent doc의 id를 가지고 있습니다. parent, child의 문서 id는 저장하여 문서 삭제, 업데이트시에 활용됩니다. 세부 코드는 [Lambda-Document](https://github.com/kyopark2014/agentic-rag/blob/main/lambda-document-manager/lambda_function.py)을 참조합니다.

```python
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    # separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
parent_docs = parent_splitter.split_documents(docs)
parent_doc_ids = vectorstore.add_documents(parent_docs, bulk_size = 10000)
ids = parent_doc_ids

for i, doc in enumerate(parent_docs):
    _id = parent_doc_ids[i]
    sub_docs = child_splitter.split_documents([doc])
    for _doc in sub_docs:
        _doc.metadata["parent_doc_id"] = _id
        _doc.metadata["doc_level"] = "child"

    child_doc_ids = vectorstore.add_documents(sub_docs, bulk_size = 10000)
    ids += child_doc_ids
```

[chat.py](https://github.com/kyopark2014/agentic-rag/blob/main/application/chat.py)와 같이 pre_filter를 이용해 child 문서를 검색하여, parent_doc_id를 이용해 parent 문서를 context로 활용합니다. 하나의 parent doc에서 여러개의 child doc이 선택될 수 있으로 parent_doc_id를 이용해 중복을 확인하여 제거합니다.

```python
result = vectorstore_opensearch.similarity_search_with_score(
    query = query,
    k = top_k*2,  
    search_type="script_scoring",
    pre_filter={"term": {"metadata.doc_level": "child"}}
)
relevant_documents = []
docList = []
for re in result:
    if 'parent_doc_id' in re[0].metadata:
        parent_doc_id = re[0].metadata['parent_doc_id']
        doc_level = re[0].metadata['doc_level']                
        if doc_level == 'child':
            if parent_doc_id in docList:
                print('duplicated!')
            else:
                relevant_documents.append(re)
                docList.append(parent_doc_id)                        
                if len(relevant_documents)>=top_k:
                    break
```

검색된 child 문서에서 parent_doc_id를 추출하여 parent 문서를 가져와 활용합니다.

```python
for i, document in enumerate(relevant_documents):
    parent_doc_id = document[0].metadata['parent_doc_id']
    doc_level = document[0].metadata['doc_level']    
    content, name, url = get_parent_content(parent_doc_id)

def get_parent_content(parent_doc_id):
    response = os_client.get(
        index = index_name, 
        id = parent_doc_id
    )    
    source = response['_source']                                
    return source['text']
```

#### Header / Footer의 제거

[Header/Footer](https://github.com/kyopark2014/ocean-agent/blob/main/lambda-document-manager/lambda_function.py)에서는 이미지의 header/footer를 제거하는 것을 보여주고 있습니다. 문서의 header/footer는 문서마다 다를 수 있으므로 문서마다 header/footer의 높이를 지정하여야 합니다.

```python
image_obj = s3_client.get_object(Bucket=s3_bucket, Key=key)
                    
image_content = image_obj['Body'].read()
img = Image.open(BytesIO(image_content))
                    
width, height = img.size     
pos = key.rfind('/')
prefix = key[pos+1:pos+5]
print('img_prefix: ', prefix)    
if pdf_profile=='ocean' and prefix == "img_":
    area = (0, 175, width, height-175)
    img = img.crop(area)
        
    width, height = img.size 
            
if width < 100 or height < 100:  # skip small size image
    return []
```

#### Multimodal을 이용해 이미지/표를 활용

문서의 이미지나 표에는 본문에 없는 중요한 정보가 있을 수 있습니다. PDF와 같은 문서에서 이미지를 추출하여 RAG에서 활용합니다. 이미지는 LLM에서 처리할 수 있도록 resize후에 텍스트를 추출합니다. 

```python
isResized = False
while(width*height > 5242880):
    width = int(width/2)
    height = int(height/2)
    isResized = True
       
if isResized:
    img = img.resize((width, height))
                     
buffer = BytesIO()
img.save(buffer, format="PNG")
img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                                                                
chat = get_multimodal()
summary = summary_image(img_base64, subject_company)
```

텍스트 추출시 아래와 같이 prompt를 이용해 이미지의 내용을 활용합니다.

```python
def summary_image(img_base64):
    chat = get_chat()
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
    
    result = chat.invoke(messages)
    return result.content
```

#### Contextual Embedding 

[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)와 같이 contextual embedding을 이용하여 chunk에 대한 설명을 추가하면, 검색의 정확도를 높일 수 있습니다. 또한 BM25(keyword) 검색은 OpenSearch의 hybrid 검색을 통해 구현할 수 있습니다. 상세한 코드는 [lambda_function.py](./lambda-document-manager/lambda_function.py)를 참조합니다.

```python
def get_contexual_docs(whole_doc, splitted_docs):
    contextual_template = (
        "<document>"
        "{WHOLE_DOCUMENT}"
        "</document>"
        "Here is the chunk we want to situate within the whole document."
        "<chunk>"
        "{CHUNK_CONTENT}"
        "</chunk>"
        "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk."
        "Answer only with the succinct context and nothing else."
        "Put it in <result> tags."
    )          
    
    contextual_prompt = ChatPromptTemplate([
        ('human', contextual_template)
    ])

    docs = []
    for i, doc in enumerate(splitted_docs):        
        chat = get_contexual_retrieval_chat()
        
        contexual_chain = contextual_prompt | chat
            
        response = contexual_chain.invoke(
            {
                "WHOLE_DOCUMENT": whole_doc.page_content,
                "CHUNK_CONTENT": doc.page_content
            }
        )
        output = response.content
        contextualized_chunk = output[output.find('<result>')+8:len(output)-9]
        
        docs.append(
            Document(
                page_content=contextualized_chunk+"\n\n"+doc.page_content,
                metadata=doc.metadata
            )
        )
    return docs
```

### Code Interpreter

"Strawberry의 'r'은 몇개인가요?"의 질문을 하면 code interpreter가 생성한 코드를 실행하여 아래와 같은 결과를 얻을 수 있습니다.

<img width="550" alt="image" src="https://github.com/user-attachments/assets/0b1f6ccd-618a-453b-8d63-f71bcf7ffa0b" />

이때 실행된 코드는 아래와 같습니다.

```python
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'
word = "Strawberry"
r_count = word.lower().count(\'r\')
print(f"\'Strawberry\'에서 \'r\'의 개수는 {r_count}개 입니다.")
```

LangSmith에서 확인한 동작은 아래와 같습니다.

![noname](https://github.com/user-attachments/assets/ef3f4d7b-620b-4257-a0b9-975598b27783)


### Agentic RAG

아래와 같이 activity diagram을 이용하여 node/edge/conditional edge로 구성되는 tool use 방식의 agent를 구현할 수 있습니다.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/59c8dc05-c79c-4f63-b1ab-964dec259203"/>


Tool use 방식 agent의 workflow는 아래와 같습니다. Fuction을 선택하는 call model 노드과 실행하는 tool 노드로 구성됩니다. 선택된 tool의 결과에 따라 cycle형태로 추가 실행을 하거나 종료하면서 결과를 전달할 수 있습니다.

```python
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

app = workflow.compile()
inputs = [HumanMessage(content=query)]
config = {
    "recursion_limit": 50
}
message = app.invoke({"messages": inputs}, config)
```

Tool use 패턴의 agent는 정의된 tool 함수의 docstring을 이용해 목적에 맞는 tool을 선택합니다. 아래의 search_by_opensearch는 OpenSearch를 데이터 저장소로 사용하여 관련된 문서를 얻어오는 tool의 예입니다. "Search technical information by keyword"로 정의하였으므로 질문이 기술적인 내용이라면 search_by_opensearch가 호출되게 됩니다.

```python
@tool    
def search_by_opensearch(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    
    
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    
    # retrieve
    relevant_docs = rag.retrieve_documents_from_opensearch(keyword, top_k=2)                            

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
       
    relevant_context = "" 
    for doc in filtered_docs:
        content = doc.page_content        
        relevant_context = relevant_context + f"{content}\n\n"
        
    return relevant_context  
```



아래와 같이 tool들로 tools를 정의한 후에 [bind_tools](https://python.langchain.com/docs/how_to/chat_models_universal_init/#using-a-configurable-model-declaratively)을 이용하여 call_model 노드를 정의합니다. 

```python
tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_knowledge_base]        

def call_model(state: State, config):
    system = (
        "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    model = chat.bind_tools(tools)
    chain = prompt | model
        
    response = chain.invoke(state["messages"])

    return {"messages": [response]}
```

또한, tool 노드는 아래와 같이 [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode)을 이용해 정의합니다.

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```


### Corrective RAG

[Corrective RAG(CRAG)](https://github.com/kyopark2014/langgraph-agent/blob/main/corrective-rag-agent.md)는 retrieval/grading 후에 질문을 rewrite한 후 인터넷 검색에서 얻어진 결과로 RAG의 성능을 강화하는 방법입니다. 

아래는 PUML로 그린 graph 입니다.

![image](https://github.com/user-attachments/assets/1526adcd-9226-4db0-9e4b-6bd01d8f6c04)

아래는 LangGraph Builder로 그린 graph 입니다. 

<img src="https://github.com/user-attachments/assets/67be90cf-7296-4381-a30c-d92b7c2638a5" width="400">


CRAG의 workflow는 아래와 같습니다. 

```python
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
```

### Self RAG

[Self RAG](https://github.com/kyopark2014/langgraph-agent/blob/main/self-rag.md)는 retrieve/grading 후에 generation을 수행하는데, grading의 결과에 따라 필요시 rewtire후 retrieve를 수행하며, 생성된 결과가 hallucination인지, 답변이 적절한지를 판단하여 필요시 rewtire / retrieve를 반복합니다. 

아래는 PUML로 그린 graph 입니다.

![image](https://github.com/user-attachments/assets/b1f2db6c-f23f-4382-86f6-0fa7d3fe0595)

아래는 LangGraph Builder로 그린 graph 입니다. 

![builder-self-rag](https://github.com/user-attachments/assets/aeb740a7-abba-451a-9c36-3f358a6653c6)


Self RAG의 workflow는 아래와 같습니다.

```python
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
```

### Self Corrective RAG

Self Corrective RAG는 Self RAG처럼 retrieve / generate 후에 hallucination인지 답변이 적절한지 확인후 필요시 질문을 rewrite하거나 인터넷 검색을 통해 RAG의 성능을 향상시키는 방법입니다. 

![image](https://github.com/user-attachments/assets/9a18f7f9-0249-42f7-983e-c5a7f9d18682)

Self Corrective RAG의 workflow는 아래와 같습니다. 

```python
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
```


### 활용 방법

EC2는 Private Subnet에 있으므로 SSL로 접속할 수 없습니다. 따라서, [Console-EC2](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)에 접속하여 "app-for-llm-streamlit"를 선택한 후에 Connect에서 sesseion manager를 선택하여 접속합니다. 

Github에서 app에 대한 코드를 업데이트 하였다면, session manager에 접속하여 아래 명령어로 업데이트 합니다. 

```text
sudo runuser -l ec2-user -c 'cd /home/ec2-user/agentic-rag && git pull'
```

Streamlit의 재시작이 필요하다면 아래 명령어로 service를 stop/start 시키고 동작을 확인할 수 있습니다.

```text
sudo systemctl stop streamlit
sudo systemctl start streamlit
sudo systemctl status streamlit -l
```

Local에서 디버깅을 빠르게 진행하고 싶다면 [Local에서 실행하기](https://github.com/kyopark2014/agentic-rag/blob/main/deployment.md#local%EC%97%90%EC%84%9C-%EC%8B%A4%ED%96%89%ED%95%98%EA%B8%B0)에 따라서 Local에 필요한 패키지와 환경변수를 업데이트 합니다. 이후 아래 명령어서 실행합니다.

```text
streamlit run application/app.py
```

EC2에서 debug을 하면서 개발할때 사용하는 명령어입니다.

먼저, 시스템에 등록된 streamlit을 종료합니다.

```text
sudo systemctl stop streamlit
```

이후 EC2를 session manager를 이용해 접속한 이후에 아래 명령어를 이용해 실행하면 로그를 보면서 수정을 할 수 있습니다. 

```text
sudo runuser -l ec2-user -c "/home/ec2-user/.local/bin/streamlit run /home/ec2-user/agentic-rag/application/app.py"
```


## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행 결과

메뉴에서는 아래와 항목들을 제공하고 있습니다.

![image](https://github.com/user-attachments/assets/53049a34-68c8-4506-8b1e-d34ca70a6a7f)


메뉴에서 [이미지 분석]과 모델로 [Claude 3.5 Sonnet]을 선택한 후에 [기다리는 사람들 사진](./contents/waiting_people.jpg)을 다운받아서 업로드합니다. 이후 "사진속에 있는 사람들은 모두 몇명인가요?"라고 입력후 결과를 확인하면 아래와 같습니다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/3e1ea017-4e46-4340-87c6-4ebf019dae4f" />



### RAG (Knowledge Base)



### Reference 

[Nova Pro User Guide](https://docs.aws.amazon.com/pdfs/nova/latest/userguide/nova-ug.pdf)

[7 Agentic RAG System Architectures](https://www.linkedin.com/posts/greg-coquillo_7-agentic-rag-system-architectures-ugcPost-7286098967434534912-F2Ek/?utm_source=share&utm_medium=member_android)

[how multi-agent agentic RAG systems work](https://www.linkedin.com/posts/pavan-belagatti_lets-understand-how-multi-agent-agentic-activity-7286068649101008896-DmDB/?utm_source=share&utm_medium=member_android)
