from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

import streamlit as st

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph.graph import Graph

## loading the api keys from the .env file
load_dotenv()
os.environ["OPENWEATHERMAP_API_KEY"] = os.environ.get("OPENWEATHERMAP_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langsmith Tracing
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"] = "weather-app"

model = ChatOpenAI(model="gpt-3.5-turbo")



## we keep a dictionary representing state that is passed along all the nodes in the graph
AgentState = {}
# messages key is assigned an empty array initially, we will add new messages as we pass along all the nodes in the graph
AgentState["messages"]=[]

# Node 1 - Extracts city name from user input using the LLM model
def function_1(state):
    messages = state['messages']
    user_input= messages[-1]

    # Construct query to extract city name from user input
    complete_query = "Your task is to provide only the city name based on the user query. \
        Nothing more, just the city name mentioned. Following is the user query: " + user_input
    response = model.invoke(complete_query)
    state['messages'].append(response.content) # appending AIMessage response(city_name) to the AgentState
    return state

# Node 2 - Fetches weather information for the location using OpenWeatherMap API
def function_2(state):
    messages = state['messages']
    agent_response = messages[-1]
    # Initialize OpenWeatherMap API wrapper and fetch weather data
    weather = OpenWeatherMapAPIWrapper()
    weather_data = weather.run(agent_response)
    state['messages'].append(weather_data)
    return state

# Node 3 - Generates temperature information in Celsius and Fahrenheit based on the weather data as well as the user input from the first node
def function_3(state):
    messages = state['messages']
    user_input = messages[0]
    agent_response = messages[-1]
    ## prompt tuning to get the temperature only (in both Celsius and Fahrenheit)
    agent2_query = "Your task is to provide weather info concisely based on the user query and the available information from the internet. Then, provide the temp in both celsius and fahrenheit \
                        Following is the user query: " + user_input + " Available information: " + agent_response
    response = model.invoke(agent2_query)
    return response.content



# Define a Langchain graph workflow
workflow = Graph()

#calling node 1(agent) - LLM model
workflow.add_node("agent", function_1)
workflow.add_node("tool", function_2) #calling node 2 (tool) - Weather API
workflow.add_node("responder", function_3) #calling node 3 (responder) - Temperature extraction

# Define the connections between nodes in the graph
workflow.add_edge('agent', 'tool')
workflow.add_edge('tool','responder')

# Set entry and finish points for the workflow
workflow.set_entry_point("agent")
workflow.set_finish_point("responder")

app = workflow.compile() # Compile the workflow

st.set_page_config(page_title="Weather Information App", page_icon="🌦️", layout="wide")
st.markdown("""
    <style>
    .reportview-container {
        background-color: #fff8ff;  
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
    }
    .stButton>button {
        background-color: #ff5566;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<p class='big-font'>🌦️ Weather Information App</p>", unsafe_allow_html=True)
st.markdown("<p class='medium-font'>Get real-time weather information for any location!</p>", unsafe_allow_html=True)

# Define layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("Enter your location", placeholder="e.g., New York, London, Tokyo")
    if st.button("Get Weather"):
        if query:
            with st.spinner("Fetching weather information..."):
                state = {"messages": [f"what is the temperature in {query}"]}
                result = app.invoke(state)
                st.success("Weather information retrieved!")
                st.markdown("### Weather Report")
                st.write(result)
        else:
            st.error("Please enter a location")

with col2:
    st.markdown("### How to use")
    st.write("1. Enter a location in the text box")
    st.write("2. Click 'Get Weather' button")
    st.write("3. Wait for the results to appear")
    
    st.markdown("### About")
    st.write("This app uses OpenAI's GPT model and OpenWeatherMap API to provide accurate and up-to-date weather information.")


# Example of streaming output from the app
input = {"messages": ["what is the temperature in Toronto"]}
for output in app.stream(input):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")

