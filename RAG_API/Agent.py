import requests
from llama_index.core.tools import FunctionTool

from tools import query_RAG_pipeline
from tools import gettemp

# This part is gpt generated , i will have to fix the depreciated warning ....

import asyncio
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context

rag_tool = FunctionTool.from_defaults(
    fn=query_RAG_pipeline,
    name="query_rag_pipeline",
    description="Searches uploaded documents and answers questions.")

temp_tool = FunctionTool.from_defaults(fn = gettemp , name = "get_temperature" , description="Returns the temperature of a city")

#Initialize the LLM (Gemini 2.5 Flash)
llm = GoogleGenAI(model="gemini-2.5-flash")

# Define the Function Agent configuration
agent_config = FunctionAgent(
    name="rag_assistant",
    tools=[rag_tool , temp_tool],
    llm=llm,
    system_prompt="""
    You are a helpful assistant.

    Use query_rag_pipeline whenever the user asks
    something that may require document knowledge.
    Use Temp tool when asked temperature of a city
    """
)

# 3. Wrap it in the AgentWorkflow orchestrator
agent = AgentWorkflow(agents=[agent_config], root_agent="rag_assistant")

# 4. Run using the standard workflow signature with Context
async def main():
    # Creating a dedicated context forces the standard workflow signature
    ctx = Context(agent)
    
    # Passing the context satisfies workflow.run(ctx=ctx, user_msg='...')
    response = await agent.run(
        ctx=ctx,
        user_msg="Tell me the temperature in london" # actually returned it .. meaning it works !!!
    )
    print(response)

# Execute the async function
asyncio.run(main())