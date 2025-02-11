import asyncio
import json
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
import gradio as gr
from crawl4ai import *
from crawl4ai.async_webcrawler import AsyncWebCrawler


# ---------------------------
# Deep researcher based on 
# https://github.com/mshumer/OpenDeepResearcher
# ---------------------------

# ---------------------------
# Requirements:
# 
# 1. Crawl4ai: https://github.com/unclecode/crawl4ai
# pip install -U crawl4ai
# 2. Create account on LangSearch
# and put your api key in .env file
# LANGSEARCH_API_KEY=sk-12345
# ---------------------------

# ---------------------------
# Configuration Constants
# ---------------------------

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
LANGSEARCH_URL = "https://api.langsearch.com/v1/web-search"

client = OpenAI()

search_periods = {
    "Day": "oneDay",
    "Week": "oneWeek",
    "Month": "oneMonth",
    "Year": "oneYear",
    "No Limit": "noLimit"
}

# -------------------------------
# Helper Functions
# -------------------------------

def call_openai(messages, model=DEFAULT_MODEL):
    print(f"{datetime.now()} Calling OpenAI")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error calling OpenAI:", e)
        return None
    
def generate_search_queries(user_query):
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather complete information on the topic. "
        "Return only a list of strings, for example: ['query1', 'query2', 'query3']."
    )
    messages = [
        {"role": "system", "content": f"You are a helpful and precise research assistant. Today is {datetime.now().date()}"},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    response = call_openai(messages)
    if response:
        try:
            search_queries = eval(response)
            if isinstance(search_queries, list):
                return search_queries
            else:
                print("LLM did not return a list. Response:", response)
                return []
        except Exception as e:
            print("Error parsing search queries:", e, "\nResponse:", response)
            return []
    return []

def perform_search(query, freshness):
    headers = {
        "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "query": query,
        "freshness": search_periods[freshness],
        "summary": False,
        "count": 10
    })
    try:
        resp = requests.request("POST", LANGSEARCH_URL, headers=headers, data=payload)
        result = resp.json()
        if result["code"] == 200:
            if "data" in result and "webPages" in result["data"] and "value" in result["data"]["webPages"]:
                return [item["url"] for item in result["data"]["webPages"]["value"]]
            else:
                print("No links in LANGSEARCH response.")
                return []
        else:
            print(f"LANGSEARCH error: {resp.status} - {resp.text}")
            return []
    except Exception as e:
        print("Error performing LANGSEARCH search:", e)
        return []
    
async def fetch_webpage_text(url):
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result.success:
                return result.markdown
            else:
                print(f"Crawl4AI fetch error for {url}: {result.error_message}")
                return ""
    except Exception as e:
        print("Error fetching webpage text with Crawl4AI:", e)
        return ""
    
def is_page_useful(user_query, page_text):
    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, "
        "determine if the webpage contains information that is useful for addressing the query. "
        "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."
    )
    messages = [
        {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
        {"role": "user", "content": f"User Query: {user_query}\n\nWebpage Content (first 5000 characters):\n{page_text[:5000]}\n\n{prompt}"}
    ]
    response = call_openai(messages)
    if response:
        answer = response.strip()
        if answer in ["Yes", "No"]:
            return answer
        else:
            if "Yes" in answer:
                return "Yes"
            elif "No" in answer:
                return "No"
    return "No"

def extract_relevant_context(user_query, search_query, page_text):
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are useful for answering the user's query. "
        "Return only the relevant context as plain text without extra commentary."
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
        {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\n\nWebpage Content (first 5000 characters):\n{page_text[:5000]}\n\n{prompt}"}
    ]
    response = call_openai(messages)
    if response:
        return response.strip()
    return ""

def get_new_search_queries(user_query, previous_search_queries, all_contexts):
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, decide if further research is needed. "
        "If further research is needed, provide up to four new search queries as a Python list (for example, "
        "['new query1', 'new query2']). If you believe no further research is needed, respond with empty string exactly ."
        "\nOutput only a Python list or the token without any extra text."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    response = call_openai(messages)
    if response:
        cleaned = response.strip()
        if cleaned == "":
            return ""
        try:
            new_queries = eval(cleaned)
            if isinstance(new_queries, list):
                return new_queries
            else:
                print("LLM did not return a list for new search queries. Response:", response)
                return []
        except Exception as e:
            print("Error parsing new search queries:", e, "\nResponse:", response)
            return []
    return []

def generate_final_report(user_query, all_contexts):
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
        "write a complete, well-structured, and detailed report that addresses the query thoroughly. "
        "Include all useful insights and conclusions without extra commentary."
    )
    messages = [
        {"role": "system", "content": "You are a skilled report writer."},
        {"role": "user", "content": f"User Query: {user_query}\n\nGathered Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    report = call_openai(messages)
    return report

def process_link(link, user_query, search_query, log):
    log.append(f"Fetching content from: {link}")
    page_text = asyncio.run(fetch_webpage_text(link))
    if not page_text:
        log.append(f"Failed to fetch content from: {link}")
        return None
    usefulness = is_page_useful(user_query, page_text)
    log.append(f"Page usefulness for {link}: {usefulness}")
    if usefulness == "Yes":
        context = extract_relevant_context(user_query, search_query, page_text)
        if context:
            log.append(f"Extracted context from {link} (first 200 chars): {context[:200]}")
            return context
    return None

# -----------------------------
# Main Routine
# -----------------------------

def research(user_query, iteration_limit, freshness):
    aggregated_contexts = []
    all_search_queries = []
    log_messages = []  # List to store intermediate steps
    iteration = 0

    log_messages.append("Generating initial search queries...")
    new_search_queries = generate_search_queries(user_query)
    if not new_search_queries:
        log_messages.append("No search queries were generated by the LLM. Exiting.")
        return "No search queries were generated by the LLM. Exiting.", "\n".join(log_messages)
    
    all_search_queries.extend(new_search_queries)
    log_messages.append(f"Initial search queries: {new_search_queries}")

    while iteration < iteration_limit:
        log_messages.append(f"\n=== Iteration {iteration + 1} ===")
        iteration_contexts = []
        search_results = []
        query_link_map = {}  # Track which query produced which links
    
        # Perform searches and store query-link mapping
        for query in new_search_queries:
            links = perform_search(query, freshness)
            search_results.extend(links)
            for link in links:
                query_link_map[link] = query  # Associate each link with its query
    
        log_messages.append(f"Aggregated {len(query_link_map)} unique links from this iteration.")
    
        for link in query_link_map:
            res = process_link(link, user_query, query_link_map[link], log_messages)
            if res:
                iteration_contexts.append(res)
    
        if iteration_contexts:
            aggregated_contexts.extend(iteration_contexts)
            log_messages.append(f"Found {len(iteration_contexts)} useful contexts in this iteration.")
        else:
            log_messages.append("No useful contexts were found in this iteration.")
    
        # Get new search queries
        new_search_queries = get_new_search_queries(user_query, all_search_queries, aggregated_contexts)
    
        if not new_search_queries:  # Handles both empty string and empty list cases
            log_messages.append("LLM indicated that no further research is needed.")
            break
        else:
            log_messages.append(f"LLM provided new search queries: {new_search_queries}")
            all_search_queries.extend(new_search_queries)
    
        iteration += 1

    log_messages.append("\nGenerating final report...")
    final_report = generate_final_report(user_query, aggregated_contexts)
    try:
        save_report(user_query, final_report)
    except Exception as e:
        print(f"Error on saving report: {e}")
    return final_report, "\n".join(log_messages)
    
def save_report(user_query, report):
    output_dir = os.path.abspath("./DeepSearcherReports")
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = determine_report_name(user_query)
    file_path = f"{output_dir}/{file_name}"
    
    with open(file_path, "w") as file:
        file.write(report)

def determine_report_name(user_query):
    prompt = (
        "Based on this user query prepare nice and concise file name that could be used ",
        "to save a report on the topic the user is interested in. ",
        "Reply with just the file name with .md extension for markdown at the end."
    )
    messages = [
        {"role": "system", "content": "You are a helpfull assistant."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    return call_openai(messages)

    
def run_research(user_query, iteration_limit, freshness):
    return research(user_query, iteration_limit, freshness)

# -----------------------------
# Gradio UI Setup
# -----------------------------

def gradio_run(user_query, iteration_limit, freshness):
    try:
        final_report, logs = run_research(user_query, int(iteration_limit), freshness)
        return final_report, logs
    except Exception as e:
        return f"An error occurred: {e}", ""
    
default_prompt = (
    "Analyze the future perspectives for gold prices in the upcoming months. "
    "Focus on macroeconomic factors (e.g., inflation, interest rates, GDP growth), "
    "political dynamics (e.g., policy shifts, geopolitical tensions) "
    "and historical gold price trends influenced by major world events. "
    "Provide a comprehensive forecast integrating these elements."
)

iface = gr.Interface(
    fn=gradio_run,
    inputs=[
        gr.Textbox(lines=5, value=default_prompt, label="Research Query/Topic"),
        gr.Number(value=4, label="Max Iterations"),
        gr.Dropdown(search_periods.keys(), value="Month", label="Search period")
    ],
    outputs=[
        gr.Markdown(label="Final Report"),
        gr.Textbox(label="Intermediate Steps Log")
    ],
    title="Research Assistant",
    description="Enter your query and a maximum iteration count to generate a report. The log will show the steps taken.",
    flagging_mode='never'
)

iface.launch()