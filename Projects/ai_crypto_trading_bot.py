import asyncio
import json
import os
import time
from crawl4ai import AsyncWebCrawler
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from binance.spot import Spot
import requests

# Create an account on https://testnet.binance.vision/,
# Generate HMAC-SHA-256 Key
# paste an API Key and API Secret in .env file
# BINANCE_API_KEY=12345
# BINANCE_API_SECRET=12345
# Documentation: https://www.binance.com/en/support/faq/how-to-test-my-functions-on-binance-testnet-ab78f9a1b8824cf0a106b4229c76496d
# Github Binance Connector: https://github.com/binance/binance-connector-python
# pip install binance-connector

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
LANGSEARCH_URL = "https://api.langsearch.com/v1/web-search"
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_TESTNET_URL = "https://testnet.binance.vision"

llm_client = OpenAI()
client = Spot(
    api_key=BINANCE_API_KEY,
    api_secret=BINANCE_API_SECRET,
    base_url=BINANCE_TESTNET_URL)

# -------------------------------
# LLM helper functions
# -------------------------------

def call_llm(messages, model=DEFAULT_MODEL):
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error calling OpenAI:", e)
        return None
    
# -------------------------------
# LangSearch Functions
# -------------------------------

def perform_search(user_query, freshness):
    headers = {
        "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "query": user_query,
        "freshness": freshness,
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
    
# -------------------------------
# Crawl4AI Functions
# -------------------------------
    
async def fetch_webpage_text(url):
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result.success:
                return result.markdown[:6000]
            else:
                print(f"Crawl4AI fetch error for {url}: {result.error_message}")
                return ""
    except Exception as e:
        print("Error fetching webpage text with Crawl4AI:", e)
        return ""
    
# -------------------------------
# Helper Functions
# -------------------------------

def check_sentiment(ticker="BTC"):
    query = f"current sentiment analysis for {ticker} cryptocurrency"
    links = perform_search(query, "oneDay")
    pages_in_context = 0
    global_context = ""
    for link in links:
        content = asyncio.run(fetch_webpage_text(link))
        is_useful = is_page_useful(query, content)
        answer = is_useful.strip()
        if "Yes" in answer:
            pages_in_context = pages_in_context + 1
            global_context += f"Content from {link}\n{content}\n"
    
    system_message = (
        "You are a critical sentiment evaluator. Given the user's query and the context from multiple web pages, "
        "determine if the current market sentiment is positive, negative or neutral. "
        "Respond with exactly one work: 'Positive' if the sentiment is positive, 'Negative' if the sentiment is negative "
        "or 'Neutral' if the sentiment is neutral. Do not include any extra text, just a single word."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"User Query: {query}\n\nContext from web pages:\n{global_context}"}
    ]
    response = call_llm(messages)
    if response:
        print(f"Pages in context: {pages_in_context}")
        return response.strip()
    
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
    response = call_llm(messages)
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

# Calculate RSI based on closing prices for the past week
def calculate_rsi(symbol='BTCUSDT', interval='15m', period=24*4):
    prices = get_market_data(symbol, interval, period)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = [100 - 100 / (1 + rs)]
    for delta in deltas[period:]:
        up_val = max(delta, 0)
        down_val = -min(delta, 0)
        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        rs = up / down if down != 0 else 0
        rsi.append(100 - 100 / (1 + rs))
    return rsi[-1]

# Get recent market data using klines; index 4 is the closing price
def get_market_data(symbol, interval, limit):
    klines = client.klines(symbol=symbol, interval=interval, limit=limit)
    return [float(kline[4]) for kline in klines]

# -------------------------------
# Decision Making
# -------------------------------

# Combine signals from sentiment and technical analysis
def decide_trade(sentiment, rsi):
    # Example decision: buy when sentiment is Positive and RSI indicates oversold (<30);
    # sell when sentiment is Negative and RSI indicates overbought (>70).
    if sentiment in ['Positive', 'Neutral'] and rsi < 30:
        return 'buy'
    elif sentiment in ['Nagative', 'Neutral'] and rsi > 70:
        return 'sell'
    return 'hodl'

# Execute trade via Binance Spot API
def execute_order(signal, symbol='BTCUSDT', quantity=0.001):
    if signal == 'buy':
        return client.new_order(symbol=symbol, side='BUY', type='MARKET', quantity=quantity)
    elif signal == 'sell':
        return client.new_order(symbol=symbol, side='SELL', type='MARKET', quantity=quantity)
    return None

# -------------------------------
# Main Workflow
# -------------------------------

# BITCOIN by default
def run_workflow():
    # 1. Get recent market data and calculate RSI (technical indicator)
    rsi = calculate_rsi()

    # 2. Analyze sentiment
    sentiment = check_sentiment()

    # 3. Decide trade based on combined signals and current position status
    decision = decide_trade(sentiment, rsi)

    print("\n********************\n")

    # 4. Execute order if decision is buy or sell
    if decision in ['buy', 'sell']:
        order = execute_order(decision)
        print(f"{datetime.now()} Order executed: {order}")
    else:
        print(f"{datetime.now()} Hodl.")

    print(f"{datetime.now()} Sentiment: {sentiment} | RSI: {rsi} | Decision: {decision}")
    print("\n********************\n")


# -------------------------------
# Scheduler
# -------------------------------

def main_loop():
    while True:
        try:
            run_workflow()
        except Exception as e:
            print(f"Error: {e}")
        # Schedule next run (e.g., every 15 minutes)
        time.sleep(900)


main_loop()