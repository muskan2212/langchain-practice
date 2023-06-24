import os
from langchain.llms import OpenAI
from langchain.document_loaders import NotionDirectoryLoader

from langchain import HuggingFaceHub, ConversationChain

from langchain.agents import load_tools
from langchain.agents import initialize_agent
import requests
from langchain.vectorstores import FAISS
# pip install wikipedia


os.environ["YOUR_OPENAI_TOKEN"]='YOUR_OPENAI_TOKEN'

llm = OpenAI(temperature=0.4, token='YOUR_OPENAI_TOKEN')
text = "What would be a good company name for a company that makes good color socks?"
print("**1**",llm(text))

#using huggingface
os.environ["huggingface_write"]="huggingface_write"

"""
    https://huggingface.co/google/flan-t5-xl
    
    is used for text to text
"""
llm =HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_lenght":64})
llm("translate English to German: How old are you?")


#to send input to llm use prompt template
x=llm("Can Barack Obama have a conversation with George Washington?")
print("***2***",x)

#prompt
prompt = """Question: Can Barack Obama have a conversation with George Washingoton?

Let's think step by step.

Answer: """

y= llm(prompt)
print("***3***",y)

#we can achived propmt template using PrompTemplates:
from  langchain import PromptTemplate

template="""Question: {question}
Let's think step by step.
Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

prompt.format(question="Can Barack Obama have a conversation with George Washington?")

#Chains->Combine LLMs and Prompts in multi-step workflows
from langchain import LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Can Barack Obama have a conversation with George Washington?"
print(llm_chain.run(question))


#agents and tools
"""
Agents involve an LLM making decisions about which cctions to take, taking that cction, 
seeing an observation, and repeating that until done.

Tool: A function that performs a specific duty. This can be things like: Google Search, Database lookup, Python REPL, other chains.
LLM: The language model powering the agent.
Agent: The agent to use.

"""

from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-short-react-description",verbose=True)
agent.run("In what year was the film Departed with Leopnardo Dicaprio released? What is this year raised to the 0.43 power?")

"""
memory 
Add state to Chains and Agents.

"""
llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)
conversation.predict(input="hi there !")
conversation.predict(input="Can we talk about AI?")
conversation.predict(input="I'm interested in Reinforcement Learning.")

 
#document loaders
loader = NotionDirectoryLoader("Notion_DB")
docs = loader.load()


"""
Indexes:
Indexes refer to ways to structure documents so that LLMs can best interact with them. This module contains utility functions for working with documents

Embeddings: An embedding is a numerical representation of a piece of information, for example, text, documents, images, audio, etc.
Text Splitters: When you want to deal with long pieces of text, it is necessary to split up that text into chunks.
Vectorstores: Vector databases store and index vector embeddings from NLP models to understand the meaning and context of strings of text, sentences, and whole documents for more accurate and relevant search results.

"""
url = "https://github.com/hwchase17/chat-your-data/blob/master/state_of_the_union.txt"
res = requests.get(url)

with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)


#document loader
from langchain.document_loaders import TextLoader
loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

#text splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs=text_splitter.split_documents(documents)


#embedding
embeddings = HuggingFaceHub()

db= FAISS.from_documents(docs,embeddings)
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

print("***4***",docs[0].page_content)

#save and load
db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings)
docs = new_db.similarity_search(query)
print("***5***",docs[0].page_content)

