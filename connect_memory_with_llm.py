import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from langchain.agents import tool

from langchain_core.runnables import RunnableConfig
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN
        
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
retriever = db.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(llm=load_llm(HUGGINGFACE_REPO_ID), retriever=retriever, chain_type_kwargs={"prompt": prompt})



def rag_node(state):
    messages = state["messages"]
    last_user_msg = messages[-1].content if messages else ""
    answer = qa_chain.invoke({"query": last_user_msg})
    return {"messages": messages + [AIMessage(content=answer["result"])]}
    #return {"messages": messages + [HumanMessage(content=answer)]}


# LangGraph graph definition
class GraphState(TypedDict):
    messages: List[BaseMessage]

graph = StateGraph(GraphState)
graph.add_node("rag", rag_node)
graph.set_entry_point("rag")
graph.set_finish_point("rag")
compiled_graph = graph.compile()

if __name__ == "__main__":
    user_input = input("Write your query here: ") #"I have chest pain and a cough"
    output = compiled_graph.invoke({"messages": [HumanMessage(content=user_input)]})
    print(output["messages"][-1].content)