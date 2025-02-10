import utils
import chat
import utils
import json
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from langchain.docstore.document import Document

logger = utils.CreateLogger("rag-opensearch")

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

opensearch_url = config["opensearch_url"] if "opensearch_url" in config else None
if opensearch_url is None:
    raise Exception ("No OpenSearch URL")

opensearch_account = config["opensearch_account"] if "opensearch_account" in config else None
if opensearch_account is None:
    raise Exception ("Not available OpenSearch!")
opensearch_passwd = config["opensearch_passwd"] if "opensearch_passwd" in config else None
if opensearch_passwd is None:
    raise Exception ("Not available OpenSearch!")

projectName = config["projectName"] if "projectName" in config else "langgraph-nova"

index_name = projectName

os_client = OpenSearch(
    hosts = [{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_compress = True,
    http_auth=(opensearch_account, opensearch_passwd),
    use_ssl = True,
    verify_certs = True,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)

def lexical_search(query, top_k):
    # lexical search (keyword)
    min_match = 0
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "minimum_should_match": f'{min_match}%',
                                "operator":  "or",
                            }
                        }
                    },
                ],
                "filter": [
                ]
            }
        }
    }

    response = os_client.search(
        body=query,
        index=index_name
    )
    # print('lexical query result: ', json.dumps(response))
        
    docs = []
    for i, document in enumerate(response['hits']['hits']):
        if i>=top_k: 
            break
                    
        excerpt = document['_source']['text']
        
        name = document['_source']['metadata']['name']
        # print('name: ', name)

        page = ""
        if "page" in document['_source']['metadata']:
            page = document['_source']['metadata']['page']
        
        url = ""
        if "url" in document['_source']['metadata']:
            url = document['_source']['metadata']['url']            
        
        docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'page': page,
                        'from': 'lexical'
                    },
                )
            )
    
    for i, doc in enumerate(docs):
        #print('doc: ', doc)
        #print('doc content: ', doc.page_content)
        
        if len(doc.page_content)>=100:
            text = doc.page_content[:100]
        else:
            text = doc.page_content            
        logger.info(f"--> lexical search doc[{i}]: {text}, metadata:{doc.metadata}")   
        
    return docs

def get_parent_content(parent_doc_id):
    response = os_client.get(
        index = index_name, 
        id = parent_doc_id
    )
    
    source = response['_source']                            
    # print('parent_doc: ', source['text'])   
    
    metadata = source['metadata']    
    #print('name: ', metadata['name'])   
    #print('url: ', metadata['url'])   
    #print('doc_level: ', metadata['doc_level']) 
    
    url = ""
    if "url" in metadata:
        url = metadata['url']
    
    return source['text'], metadata['name'], url

def retrieve_documents_from_opensearch(query, top_k):
    logger.info(f"###### retrieve_documents_from_opensearch ######")

    # Vector Search
    bedrock_embedding = chat.get_embedding()       
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = index_name,
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    )  
    
    relevant_docs = []
    if chat.enableParentDocumentRetrival == 'enable':
        result = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k*2,  
            search_type="script_scoring",
            pre_filter={"term": {"metadata.doc_level": "child"}}
        )
        logger.info(f"result: {result}")
                
        relevant_documents = []
        docList = []
        for re in result:
            if 'parent_doc_id' in re[0].metadata:
                parent_doc_id = re[0].metadata['parent_doc_id']
                doc_level = re[0].metadata['doc_level']
                logger.info(f"doc_level: {doc_level}, parent_doc_id: {parent_doc_id}")
                        
                if doc_level == 'child':
                    if parent_doc_id in docList:
                        logger.info(f"duplicated")
                    else:
                        relevant_documents.append(re)
                        docList.append(parent_doc_id)                        
                        if len(relevant_documents)>=top_k:
                            break
                                    
        # print('relevant_documents: ', relevant_documents)    
        for i, doc in enumerate(relevant_documents):
            if len(doc[0].page_content)>=100:
                text = doc[0].page_content[:100]
            else:
                text = doc[0].page_content            
            logger.info(f"--> vector search doc[{i}]: {text}, metadata:{doc[0].metadata}")

        for i, document in enumerate(relevant_documents):
                logger.info(f"## Document(opensearch-vector) {i+1}: {document}")
                
                parent_doc_id = document[0].metadata['parent_doc_id']
                doc_level = document[0].metadata['doc_level']
                #print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
                
                content, name, url = get_parent_content(parent_doc_id) # use pareant document
                #print(f"parent_doc_id: {parent_doc_id}, doc_level: {doc_level}, url: {url}, content: {content}")
                
                relevant_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            'name': name,
                            'url': url,
                            'doc_level': doc_level,
                            'from': 'vector'
                        },
                    )
                )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k
        )
        
        for i, document in enumerate(relevant_documents):
            logger.info(f"## Document(opensearch-vector) {i+1}: {document}")   
            name = document[0].metadata['name']
            url = document[0].metadata['url']
            content = document[0].page_content
                   
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'from': 'vector'
                    },
                )
            )
    # print('the number of docs (vector search): ', len(relevant_docs))

    # Lexical Search
    if chat.enableHybridSearch == 'true':
        relevant_docs += lexical_search(query, top_k)    

    return relevant_docs

