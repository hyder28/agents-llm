from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

file_path = "report.pdf"

loader = PyPDFLoader(file_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
docs = splitter.split_documents(documents)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

file_path = "report.pdf"

loader = PyPDFLoader(file_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 300)
docs = splitter.split_documents(documents)

persist_directory = "temp"
collection_name = "annual_report"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents = docs, 
    embedding = embeddings,
    persist_directory = persist_directory,
    collection_name = collection_name
)

retriever = vectorstore.as_retriever(
    search_type = "similarity", 
    search_kwargs = {"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool retrieves relevant information from the RAG system based on the user's query.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information"

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}: \n {doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]
llm = llm.bind_tools(tools)

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from operator import add

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]

from langchain_core.messages import SystemMessage, ToolMessage

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

def node_call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state"""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)

    return {'messages': [message]}

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def node_take_action(state: AgentState) -> AgentState:
    """Execute tool calls requested by the LLM."""
    tool_calls = state["messages"][-1].tool_calls

    results = []
    for t in tool_calls:
        result = tools_dict[t['name']].invoke(t['args'].get('query',''))

        results.append(ToolMessage(tool_call_id = t['id'], name = t['name'], content = str(result)))

    return {"messages": results}

def should_continue(state: AgentState):
    """Check if the most recent LLM message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("llm", node_call_llm)
workflow.add_node("retriever_agent", node_take_action)

workflow.add_edge("retriever_agent", "llm")

workflow.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})

workflow.set_entry_point("llm")

app = workflow.compile()

# png_graph_data = app.get_graph().draw_mermaid_png()
# with open("agentic_rag.png", "wb") as f:
#     f.write(png_graph_data)

from langchain_core.messages import HumanMessage

user_input = "What’s the outlook for equities in 2024?"

# Start the agent's message state with the user's question
messages = [HumanMessage(content=user_input)]

# Invoke the workflow with the initial state
result = app.invoke({"messages": messages})

# Print the final answer returned by the agent
print("ANSWER:\n", result['messages'][-1].content)
