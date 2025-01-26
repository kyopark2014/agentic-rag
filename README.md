# Agentic RAG 구현하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Fagentic-rag&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false")](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>

여기에서는 RAG의 성능 향상 기법인 Agentic RAG, Corrective RAG, Self RAG를 구현하는 방법을 설명합니다. 또한 RAG의 데이터 수집에 필요한 PDF의 header/footer의 처리, 이미지의 추출 및 분석과 함께 contextual retrieval을 활용하는 방법을 설명합니다. 이를 통해서 생성형 AI 애플리케이션을 위한  데이터를 효과적으로 수집하여 활용할 수 있습니다. 여기서는 오픈소스 LLM Framework인 [LangGraph](https://langchain-ai.github.io/langgraph/)을 이용하고, 구현된 workflow들은 [Streamlit](https://streamlit.io/)을 이용해 개발 및 테스트를 수행할 수 있습니다. [AWS CDK](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-construct-library.html)를 이용하고 한번에 배포할 수 있고, [CloudFront](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Introduction.html) - ALB 구조를 이용해 HTTPS로 안전하게 접속할 수 있습니다. 

## System Architecture 

전체적인 architecture는 아래와 같습니다. Streamlit이 설치된 EC2는 private subnet에 있고, CloudFront-ALB를 이용해 외부와 연결됩니다. RAG는 OpenSearch를 활용하고 있습니다. 인터넷 검색은 tavily를 사용하고 날씨 API를 추가로 활용합니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/3353ade1-db6e-4d30-baa5-be78d9820418" />


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


### Corrective RAG

[Corrective RAG(CRAG)](https://github.com/kyopark2014/langgraph-agent/blob/main/corrective-rag-agent.md)는 retrival/grading 후에 질문을 rewrite한 후 인터넷 검색에서 얻어진 결과로 RAG의 성능을 강화하는 방법입니다. 

![image](https://github.com/user-attachments/assets/27228159-b307-4588-8a8a-61d8deaa90e3)

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

![image](https://github.com/user-attachments/assets/b1f2db6c-f23f-4382-86f6-0fa7d3fe0595)

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

### Contextual Embedding 

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

#### Case 1: 기업의 지분율

아래의 경우는 기업의 지분율에 대한 데이터로 아래 chunk에는 단순히 지분율 열거하고 있습니다.

```text
structure as of 3 January 2024 (date of last disclosure) is as follows:
Suzano Holding SA, Brazil - 27.76%  
David Feffer - 4.04%  
Daniel Feffer - 3.63%  
Jorge Feffer - 3.60%  
Ruben Feffer - 3.54%  
Alden Fundo De Investimento Em Ações, Brazil - 1.98%  
Other investors hold the remaining 55.45%
Suzano Holding SA is majority-owned by the founding Feffer family
Ultimate Beneficial Owners
and/or Persons with Significant
ControlFilings show that the beneficial owners/persons with significant control
are members of the Feffer family, namely David Feffer, Daniel Feffer,
Jorge Feffer, and Ruben Feffer
Directors Executive Directors:  
Walter Schalka - Chief Executive Officer  
Aires Galhardo - Executive Officer - Pulp Operation  
Carlos Aníbal de Almeida Jr - Executive Officer - Forestry, Logistics and
Procurement  
Christian Orglmeister - Executive Officer - New Businesses, Strategy, IT,
Digital and Communication
```

아래는 contexualized chunk입니다. 원본 chunk에 없는 회사명과 ownership에 대한 정보를 포함하고 있습니다.

```text
This chunk provides details on the ownership structure and key executives of Suzano SA, 
the company that is the subject of the overall document.
It is likely included to provide background information on the company's corporate structure and leadership.
```

#### Case 2: 기본의 finanacial information

아래는 어떤 기업의 financial 정보에 대한 chunk 입니다.

```text
Type of Compilation Consolidated Consolidated Consolidated
Currency / UnitsBRL ‘000 (USD 1 =
BRL 5.04)BRL ‘000 (USD 1 =
BRL 5.29)BRL ‘000 (USD 1 =
BRL 5.64)
Turnover 29,384,030 49,830,946 40,965,431
Gross results 11,082,919 25,009,658 20,349,843
Depreciation (5,294,748) (7,206,125) (6,879,132)
Operating profit (loss) 9,058,460 22,222,781 18,180,191
Interest income 1,215,644 967,010 272,556
Interest expense (3,483,674) (4,590,370) (4,221,301)
Other income (expense) 3,511,470 6,432,800 (9,347,234)
Profit (loss) before tax 12,569,930 8,832,957 (17,642,129)
Tax (2,978,271) (197,425) (6,928,009)
Net profit (loss) 9,591,659 23,394,887 8,635,532
Net profit (loss) attributable to
minorities/non-controlling
interests14,154 13,270 9,146
Net profit (loss) attributable to the
company9,575,938 23,119,235 8,751,864
Long-term assets 103,391,275 96,075,318 84,872,211
Fixed assets 57,718,542 50,656,634 38,169,703
Goodwill and other intangibles 14,877,234 15,192,971 16,034,339
```
아래는 contexualized chunk입니다. chunk에 없는 회사명을 포함한 정보를 제공합니다.

```text
This chunk provides detailed financial information about Suzano SA, 
including its turnover, gross results, operating profit, net profit, and asset details. 
It is part of the overall assessment and rating of Suzano SA presented in the document.
```

#### Case 3: 전화번호

아래는 회사 연락처에 대한 chunk입니다. 

```text
|Telephone|+55 11 3503&amp;#45;9000|
|Email|ri@suzano.com.br|
|Company Details||
|Company Type|Publicly Listed|
|Company Status|Operating|
|Sector|Industrial|
|Place of Incorporation|Brazil|
|Region of Incorporation|Bahia|
|Date of Incorporation|17 December 1987|
|Company Registered Number|CNPJ (Tax Id. No.): 16.404.287/0001&amp;#45;55|
```
이때의 contexualized chunk의 결과는 아래와 같습니다. chunk에 없는 회사의 연락처에 대한 정보를 제공할 수 있습니다.

```text
This chunk provides detailed company information about Suzano SA,
including its contact details, company type, status, sector, place and date of incorporation, and registered number.
This information is part of the overall assessment and rating of Suzano SA presented in the document.
```

### 이미지에서 텍스트 추출

이미지는 LLM에서 처리할 수 있도록 resize후에 텍스트를 추출합니다. 이때 LLM이 문서의 내용을 추출할 수 있도록 회사명등을 이용해 정보를 제공합니다.

```python
def store_image_for_opensearch(key, page, subject_company, rating_date):
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
        print(f"(croped) width: {width}, height: {height}, size: {width*height}")
                
    if width < 100 or height < 100:  # skip small size image
        return []
                
    isResized = False
    while(width*height > 5242880):
        width = int(width/2)
        height = int(height/2)
        isResized = True
        print(f"(resized) width: {width}, height: {height}, size: {width*height}")
           
    try:             
        if isResized:
            img = img.resize((width, height))
                             
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                                                                        
        # extract text from the image
        chat = get_multimodal()
        text = extract_text(chat, img_base64, subject_company)
        extracted_text = text[text.find('<result>')+8:len(text)-9] # remove <result> tag
        
        summary = summary_image(chat, img_base64, subject_company)
        image_summary = summary[summary.find('<result>')+8:len(summary)-9] # remove <result> tag
        
        if len(extracted_text) > 30:
            contents = f"[이미지 요약]\n{image_summary}\n\n[추출된 텍스트]\n{extracted_text}"
        else:
            contents = f"[이미지 요약]\n{image_summary}"
        print('image contents: ', contents)

        docs = []        
        if len(contents) > 30:
            docs.append(
                Document(
                    page_content=contents,
                    metadata={
                        'name': key,
                        'url': path+parse.quote(key),
                        'page': page,
                        'subject_company': subject_company,
                        'rating_date': rating_date
                    }
                )
            )         
        print('docs size: ', len(docs))
        
        return add_to_opensearch(docs, key)
    
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                
        
        return []
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

[7 Agentic RAG System Architectures](https://www.linkedin.com/posts/greg-coquillo_7-agentic-rag-system-architectures-ugcPost-7286098967434534912-F2Ek/?utm_source=share&utm_medium=member_android)

[how multi-agent agentic RAG systems work](https://www.linkedin.com/posts/pavan-belagatti_lets-understand-how-multi-agent-agentic-activity-7286068649101008896-DmDB/?utm_source=share&utm_medium=member_android)
