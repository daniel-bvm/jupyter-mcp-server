import json
import requests # type: ignore

messages = []

while True:
    user_input = input("User: ")
    if user_input == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    url = "http://localhost:8000/prompt"
    payload = {
        "messages": messages,
        "id": "123",
    }

    current_chunk = ""
    response = requests.post(url, json=payload, stream=True)
    response_content = ""
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            chunk_str = chunk.decode("utf-8")
            if chunk_str.startswith("data: "):
                current_chunk = chunk_str.split("data: ")[1].strip()
            else:
                current_chunk = current_chunk + chunk_str
                        
            if current_chunk == "[DONE]":
                break
            
            try:
                data = json.loads(current_chunk)
                # print(data["choices"][0]["delta"]["content"], end="", flush=True)
                response_content += data["choices"][0]["delta"]["content"]
            except json.JSONDecodeError:
                pass
                # print(chunk_str, end="\n", flush=True)

    response_content = response_content[response_content.rfind("</think>") + len("</think>"):] if "</think>" in response_content else response_content

    messages.append({"role": "assistant", "content": response_content})
    print("AI:", response_content)
