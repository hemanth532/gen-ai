from langgraph.graph import StateGraph  
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Step 1
def research(state):
    query = state["query"]
    prompt = f"Find out the top competitors for Tesla: {query}"
    print("Research is Done")

    return {"data": call_llm(prompt)}

#Step 2
def analyze(state):
    prompt = f"Analyze the strengths and weaknesses of: {state['data']}"
    print("Analysis is Done")
    return {"analysis": call_llm(prompt)}

# Step 3
def summarize(state):
    prompt = f"Summarize the market trends for: {state['analysis']} , present the output in a structured format"
    print("Summarization is Done")
    return {"summary": call_llm(prompt)}


graph = StateGraph(dict)

graph.add_node("Research", research)
graph.add_node("Analyze", analyze)
graph.add_node("Summarize", summarize)

graph.set_entry_point("Research")
graph.add_edge("Research", "Analyze")
graph.add_edge("Analyze", "Summarize")

app = graph.compile()

result = app.invoke({"query": "Cars"})



{
  "agent": "analyst",
  "task": "sales_analysis",
  "region": "West",
  "sales_drop": "18%",
  "recommendation": "increase_campaigns"
}

# Use JSON
# AGent-> research -> Analyze -> Summarize

# Resume 