# Nova로 Agentic RAG 구현하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Fagentic-rag&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false")](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>

Nova 모델는 동급 모델대비 빠르고, 높은 가성비와 함께 훌륭한 멀티모달 성능을 가지고 있습니다. Nova를 이용해서 RAG의 성능 향상 기법인 Agentic RAG, Corrective RAG, Self RAG를 구현하는 방법을 설명합니다. 또한 RAG의 데이터 수집에 필요한 PDF의 header/footer의 처리, 이미지의 추출 및 분석과 함께 contextual retrieval을 활용하는 방법을 설명합니다. 이를 통해서 생성형 AI 애플리케이션을 위한  데이터를 효과적으로 수집하여 활용할 수 있습니다. 여기서는 오픈소스 LLM Framework인 [LangGraph](https://langchain-ai.github.io/langgraph/)을 이용하고, 구현된 workflow들은 [Streamlit](https://streamlit.io/)을 이용해 개발 및 테스트를 수행할 수 있습니다. [AWS CDK](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-construct-library.html)를 이용하고 한번에 배포할 수 있고, [CloudFront](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Introduction.html) - ALB 구조를 이용해 HTTPS로 안전하게 접속할 수 있습니다. 

## System Architecture 

전체적인 architecture는 아래와 같습니다. Streamlit이 설치된 EC2는 private subnet에 있고, CloudFront-ALB를 이용해 외부와 연결됩니다. RAG는 OpenSearch를 활용하고 있습니다. 인터넷 검색은 tavily를 사용하고 날씨 API를 추가로 활용합니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/01e722bc-18ac-4d99-9905-1305f35fc2b6" />


## 상세 구현

Agentic workflow (tool use)는 아래와 같이 구현할 수 있습니다. 상세한 내용은 [chat.py](./application/chat.py)을 참조합니다.

### Basic Chat

일반적인 대화는 아래와 같이 stream으로 결과를 얻을 수 있습니다. 여기에서는 LangChain의 ChatBedrock과 Nova Pro의 모델명인 "us.amazon.nova-pro-v1:0"을 활용하고 있습니다.

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

### RAG

여기에서는 RAG 구현을 위하여 Amazon Bedrock의 knowledge base를 이용합니다. Amazon S3에 필요한 문서를 올려놓고, knowledge base에서 동기화를 하면, OpenSearch에 문서들이 chunk 단위로 저장되므로 문서를 쉽게 RAG로 올리고 편하게 사용할 수 있습니다. 또한 Hiearchical chunk을 이용하여 검색 정확도를 높이면서 필요한 context를 충분히 제공합니다. 


LangChain의 [AmazonKnowledgeBasesRetriever](https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.bedrock.AmazonKnowledgeBasesRetriever.html)을 이용하여 retriever를 등록합니다. 

```python
from langchain_aws import AmazonKnowledgeBasesRetriever

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=knowledge_base_id, 
    retrieval_config={"vectorSearchConfiguration": {
        "numberOfResults": top_k,
        "overrideSearchType": "HYBRID"   
    }},
    region_name=bedrock_region
)
```

Knowledge base로 조회하여 얻어진 문서를 필요에 따라 아래와 같이 재정리합니다. 이때 파일 경로로 사용하는 url은 application에서 다운로드 가능하도록 CloudFront의 도메인과 파일명을 조화합여 생성합니다.

```python
documents = retriever.invoke(query)
for doc in documents:
    content = ""
    if doc.page_content:
        content = doc.page_content    
    score = doc.metadata["score"]    
    link = ""
    if "s3Location" in doc.metadata["location"]:
        link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""        
        pos = link.find(f"/{doc_prefix}")
        name = link[pos+len(doc_prefix)+1:]
        encoded_name = parse.quote(name)
        link = f"{path}{doc_prefix}{encoded_name}"        
    elif "webLocation" in doc.metadata["location"]:
        link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
        name = "WEB"
    url = link
            
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
```        

얻어온 문서가 적절한지를 판단하기 위하여 아래와 같이 prompt를 이용해 관련도를 평가하고 [structured output](https://github.com/kyopark2014/langgraph-agent/blob/main/structured-output.md)을 이용해 결과를 추출합니다.

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

### Agentic RAG

아래와 같이 activity diagram을 이용하여 node/edge/conditional edge로 구성되는 tool use 방식의 agent를 구현할 수 있습니다.

<img width="261" alt="image" src="https://github.com/user-attachments/assets/31202a6a-950f-44d6-b50e-644d28012d8f" />

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

Tool use 패턴의 agent는 정의된 tool 함수의 docstring을 이용해 목적에 맞는 tool을 선택합니다. 아래의 search_by_knowledge_base는 OpenSearch를 데이터 저장소로 사용하는 knowledbe base로 부터 관련된 문서를 얻어오는 tool의 예입니다. "Search technical information by keyword"로 정의하였으므로 질문이 기술적인 내용이라면 search_by_knowledge_base가 호출되게 됩니다.

```python
@tool    
def search_by_knowledge_base(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    
    
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID" 
            }},
        )        
        docs = retriever.invoke(keyword)
    
    relevant_context = ""
    for i, doc in enumerate(docs):
        relevant_context += doc.page_content + "\n\n"        
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


### RAG (Knowledge Base)



### Reference 

[Nova Pro User Guide](https://docs.aws.amazon.com/pdfs/nova/latest/userguide/nova-ug.pdf)
