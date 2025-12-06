import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


model = (
    "gpt-4o-mini"  # gpt-4o, gpt-4o-ll, gpt-4o-mini, gpt-3.5-turbo, gpt-3.5-turbo-0613
)

# Load OpenAI key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# define model string
model = (
    "gpt-3.5-turbo"  # gpt-4o, gpt-4o-ll, gpt-4o-mini, gpt-3.5-turbo, gpt-3.5-turbo-0613
)

llm = ChatOpenAI(
    model=model,
    temperature=0,
    max_retries=5,
)
