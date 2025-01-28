# from ollama import chat
# from ollama import ChatResponse

# response: ChatResponse = chat(model='partai/dorna-llama3', messages=[
#   {
#     'role': 'user',
#     'content': 'چرا آسمان آبی است؟',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)


# from llama_index.llms.ollama import Ollama,
# from llama_index.agent.openai import OpenAIAgent


# llm = Ollama(model="partai/dorna-llama3", request_timeout=60.0)
# llm.system_prompt='pls just use yes or no'
# Ollama
# # agent = OpenAIAgent.from_tools(
# #             tools=[],
# #             llm=llm,
# #             verbose=True,
# #             # system_prompt=self.system_prompt
# #         )

# # print(llm._extend_prompt('pls use in  first of answer hello'))
# # llm.
# response = llm.complete(prompt="pls just use yes or no in belw question")
# # response = llm.complete(prompt="are you a llm model")
# # llm.chat(messages='ddd'kwargs=)
# # response = llm.stream_complete(prompt=')
# print(response)
# # print(llm.system_prompt)
# from function_calling import llm_config
# from llama_index.core.agent import ReActAgent

# from llama_index.llms.ollama import Ollama
# from llama_index.llms.ollama.base import ChatMessage, MessageRole
# from llama_index.core import PromptTemplate

# # import llama_index.llms.openai.base
# # Initialize the Ollama instance
# ollama_llm = Ollama(model="llama2", request_timeout=60.0)

# react_agent = ReActAgent.from_tools(
#     [],
#     llm=ollama_llm,
#     verbose=True,
#     allow_parallel_tool_calls=True,
# )
# # Define a system-wide prompt
# system_prompt = "pls start with hello and answer with persian lang and answer one of this  choice paris tehran newyourk"
# prompt = PromptTemplate(system_prompt)
# react_agent.update_prompts(
#         {"agent_worker:system_prompt": prompt})

# react_agent.reset()
# prompt_dict = react_agent.get_prompts()
# for k, v in prompt_dict.items():
#     print(f"Prompt: {k}\n\nValue: {v.template}")


# # Use the system-wide prompt in every chat session
# def send_message(user_message: str, chat_history=[]):
#     response = react_agent.chat(user_message,chat_history)
#     if not chat_history:
#         chat_history.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
    
#     chat_history.append(ChatMessage(role=MessageRole.USER, content=user_message))
    
#     return (response,chat_history)

# # Example usage
# chat_history = []
# response1,hist = send_message("پایتخت فرانسه کجاست؟", chat_history)
# # print("Response 1:", response1)

# response2,hist = send_message("جمعیت اونجا چقدره؟", chat_history)
# # print("Response 2:", response2)
# del react_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

context_str='''
پایتخت فرانسه پاریس 
'''
template = """بر اساس اطلاعات موجود به پرسش پاسخ دهید.\n\nاطلاعات موجود:{context}\n\nسوال: {question}\n"""

prompt = ChatPromptTemplate.from_template(context_str)

model = OllamaLLM(model="partai/dorna-llama3",prompt=prompt)

chain = prompt | model

print(chain.invoke({"question":"پایتخت فرانسه کجاست؟","context": context_str}))

# llm = OllamaLLM(
#     model='',
#     system=SYSTEM_PROMPT,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
#     )

# prompt = ChatPromptTemplate(
#     messages=[
#         # The system prompt is now sent directly to llama instead of putting it here
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("\n{user_input}\n" + AI_NAME + ":"),
#         # 
#     ]

# )

# memory = ConversationSummaryBufferMemory(
#         human_prefix=USER_NAME,
#         ai_prefix=AI_NAME,
#         llm=llm,
#         memory_key="chat_history", 
#         return_messages=True, 
#         max_token_limit=7500)

# conversation = ConversationChain(
#     prompt=prompt,
#     input_key="user_input",
#     llm=llm,
#     verbose=True,
#     memory=memory,
# # )
