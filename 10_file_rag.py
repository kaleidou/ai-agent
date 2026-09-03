from langchain_community.document_loaders import TextLoader

loader = TextLoader("knowledge.txt")

documents = loader.load()

print(documents)