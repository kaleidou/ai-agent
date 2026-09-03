from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

documents = [
    Document(
        page_content="LangChain is a framework for building applications powered by language models."
    ),
    Document(
        page_content="LangGraph is a framework for building stateful, multi-step AI applications using graphs."
    ),
    Document(
        page_content="RAG combines information retrieval with language generation."
    )
]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)
chunks = text_splitter.split_documents(documents)
#for chunk in chunks:
    #print(chunk.page_content)
    #print("----")

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

results = vector_store.similarity_search(
    "What is LangGraph",
    k=2
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
results = retriever.invoke("What is LangGraph")

for doc in results:
    print(doc.page_content)
    print("-----")

context = "\n".join(doc.page_content for doc in results)

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using only the provided context."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])
model = ChatOpenAI(model="gpt-5-mini")
chain = prompt|model
question = "What is LangGraph?"

response = chain.invoke({
    "context": context,
    "question": question
})

print(response.content)

from langchain_core.runnables import RunnablePassthrough
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
)
response = rag_chain.invoke(
    question)

print(response.content)