import asyncio
import aiohttp

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

async def call_ollama_async(session, messages, model=MODEL):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    try:
        async with session.post(OLLAMA_API, headers=HEADERS, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result['message']['content']
            else:
                text = await resp.text()
                print(f"Ollama model error: {resp.status} - {text}")
                return None
    except Exception as e:
        print("Error calling Ollama:", e)
        return None

async def main():
    system_message = "You are a helpful AI assistant. You are here to help the user with any questions they might have. If you don't know the answer, just say so."
    user_message = "How to get from Times Square to Central Park?"
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]
    async with aiohttp.ClientSession() as session:
        content = await call_ollama_async(session, messages)
        print(content)

if __name__ == '__main__':
    asyncio.run(main())
