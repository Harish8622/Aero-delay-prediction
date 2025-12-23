from flask import Flask, request, render_template_string
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agent.customer_agent import build_agent_app
from agent.helpers.prompts import agent_node_prompt


app = build_agent_app()

web_app = Flask(__name__)

conversation_state = {"messages": [SystemMessage(content=agent_node_prompt)]}

PAGE = """
<!doctype html>
<title>Aero Delay Chatbot</title>
<style>
  body { font-family: Arial, sans-serif; margin: 40px auto; max-width: 720px; }
  h1 { font-size: 20px; }
  textarea { width: 100%; height: 120px; }
  .msg { margin: 12px 0; }
  .bot { background: #f6f6f6; padding: 12px; border-radius: 6px; }
  .user { background: #e9f3ff; padding: 12px; border-radius: 6px; }
</style>
<h1>Aero Delay Chatbot</h1>
<form method="post">
  <textarea id="message" name="message" placeholder="Ask about flight delays..."></textarea>
  <br><br>
  <button type="submit">Send</button>
</form>
<script>
  const box = document.getElementById("message");
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      box.form.submit();
    }
  });
</script>
{% if history %}
  {% for item in history %}
    <div class="msg {{ item.role }}">
      <strong>{{ item.label }}:</strong> {{ item.text }}
    </div>
  {% endfor %}
{% endif %}
"""


@web_app.route("/", methods=["GET", "POST"])
def chat():
    reply = ""
    history = []
    if request.method == "POST":
        msg = request.form.get("message", "").strip()
        if msg:
            conversation_state["messages"].append(HumanMessage(content=msg))
            result_state = app.invoke(conversation_state)
            response = result_state["messages"][-1]
            reply = response.content
            conversation_state["messages"] = result_state["messages"]
    for message in conversation_state["messages"]:
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "label": "You", "text": message.content})
        elif isinstance(message, AIMessage):
            history.append({"role": "bot", "label": "AI", "text": message.content})
    history = history[-10:]
    return render_template_string(PAGE, reply=reply, history=history)


if __name__ == "__main__":
    web_app.run(host="127.0.0.1", port=8000, debug=True)
