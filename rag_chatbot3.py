# conda activate chatbot_env2

# Load documents
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WebBaseLoader
#loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#loader = OnlinePDFLoader("https://iopscience.iop.org/article/10.3847/2041-8213/ab7304/pdf")
#loader = WebBaseLoader("https://iopscience.iop.org/article/10.3847/2041-8213/ab7304")
loader = TextLoader("./apjl_letter.txt")

# Split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
splits = text_splitter.split_documents(loader.load())


# Embed and store splits

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Prompt 
# https://smith.langchain.com/hub/rlm/rag-prompt

from langchain import hub
rag_prompt = hub.pull("rlm/rag-prompt")

# LLM


from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

###
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)

###

from langchain.chains import ConversationalRetrievalChain


retriever = vectorstore.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)


while True:
    my_input = input("\n\n **** New Prompt ****:\n")
    if (my_input == "exit"):
        break
    else:
	    print(qa(my_input)["answer"])
	    print("\n")
	    print(qa(my_input)["chat_history"])
#print(qa("what are loading factors?"))
#print(qa("what are the limitations of cosmological simulations?"))