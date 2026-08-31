import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5-mini"
)
#prompt = "Explain what LangChain is in one sentence."

from langchain_core.messages import SystemMessage, HumanMessage
messages = [
    SystemMessage(content="You are a helpful Python tutor."),
    HumanMessage(content="What is a Python dictionary?")
    ]
#response = model.invoke(messages)
#print(response.content)
#messages.append(response)
#messages.append(HumanMessage(content="Give me one simple example."))
#response2 = model.invoke(messages)
#print(response2.content)

from langchain_core.prompts import ChatPromptTemplate


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {topic} tutor."),
    ("human", "Explain {question}")
])
prompt_value = prompt_template.invoke({
    "topic" : "Python",
    "question" : "What is a list comprehension?"
})
#print(prompt_value)
#response3 = model.invoke(prompt_value)
#print(response3.content)
#chain = prompt_template| model

#print(response4.content)

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
chain = prompt_template|model|parser
response4 = chain.invoke({
    "topic": "Python",
    "question" : "What is a lambda function in short"
}) 
print (response4)