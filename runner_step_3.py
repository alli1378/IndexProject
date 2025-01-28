from utils import create_embedding,pre_process
import lancedb
import numpy as np
import pyarrow as pa

db = lancedb.connect("./movies_db")
tbl = db.open_table("About_table")

# get input
user_input_text = input('please enter your question:  ')

# convert question to embedding
pre_process_data = pre_process(user_input_text)
question_embeding = create_embedding(pre_process_data)
query_vector = np.array(question_embeding[0]) 

# search with similarity
df = tbl.search(query_vector,vector_column_name="embedding")
context_dict=df.limit(50).to_pandas()['About'].to_dict()

# Convert the given data to a string.
context_str = ''
for i in context_dict:
    context_str += str(i)+': '+context_dict[i]+'\n'
# ollama
template_fa = f"""بر اساس اطلاعات موجود پنج تااز مرتبط ترین موارد نسبت به سوال  را برگردانید.\n\nاطلاعات موجود:{context_str}\n\nسوال: {user_input_text}\n"""
from ollama import chat
from ollama import ChatResponse
response: ChatResponse = chat(model='partai/dorna-llama3', messages=[
  {
    'role': 'user',
    'content': template_fa,
  },
])
# present
print(response['message']['content'])

