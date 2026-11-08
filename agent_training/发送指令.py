import requests



def ask(messages):
    r = requests.post("http://localhost/;11434/api/chat",
    json={
        "model":"qwen3.5:9b",
        "message":messages,
        "stream":False,
        "think":False,
        "options":{"num_ctx":8192},
    }
    )
    return r.json()["message"]["content"]

print(ask([{"role":"user","cotent":"你好，我是金宝"}]))
print(ask([{"role":"user","cotent":"你好，你知道我是谁吗？"}]))

#history = []
#history.append({"role":"user","cotent":"你好，我是金宝"})
