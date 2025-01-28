# import utility
from transformers import AutoTokenizer, AutoModel
import torch
from elasticsearch import Elasticsearch
import lancedb
from hazm import Normalizer,Stemmer,Lemmatizer,word_tokenize
import numpy as np
import pyarrow as pa
# hazme objects for preprocesing
stemmer = Stemmer()
lemmatizer = Lemmatizer()
normalizer = Normalizer()

# create connection to lance db
db=lancedb.connect('./movies_db')

# create object of elastic search
client = Elasticsearch("http://localhost:9200/")

# get moveis from elasticsearch
movies = client.search(index='data3',size=1410)['hits']['hits']

# create emmbeding utils
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
model = AutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
# last data
data = []

schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), list_size=768)),
        pa.field("About", pa.string()),
    ]
)
# step one
def convert_english_to_persian(text):
    '''
        This def convert persian number to english number
        input: text type str
        output: converted text type str
    '''
    persian_digit = '۰۱۲۳۴۵۶۷۸۹'
    english_digit = '0123456789'
    convert_number = str.maketrans(persian_digit,english_digit)
    return text.translate(convert_number)

def normalize_text(text):
    '''
        This def normalized text data
        we use convert_english_to_persian in this function for more normalized
        input: text type str
        output: normalized text type str
    '''
    normalized_text = normalizer.normalize(text)
    return convert_english_to_persian(normalized_text)
# step tow
def lemmatize_text(tokens):
    '''
        This def lemmatize tokens data
        input: tokens type list of str
        output: lemmatize tokens type list of str
    
    '''
    return [lemmatizer.lemmatize(token) for token in tokens]
# step one and tow
def pre_process(text):
    '''
        This def proces functions in step one and tow
        we use "normalize_text","lemmatize_text" in this def
        input: text type str
        output: preproces tokens type list of str
    '''
    normal_text = normalize_text(text)
    tokens = word_tokenize(normal_text)
    lemmatiz_tokens = lemmatize_text(tokens)
    return  lemmatiz_tokens
# step three
def create_embedding(tokens):
    '''
        This def create emmbeding from tokens 
        we have a len for length of tokens list to normal size of all emmbedings and convert tensor emmbed to list 
        input: tokens type list of str
        output: preproces tokens type list of embeds
    '''
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    max_length = 60
    if len(token_ids) < max_length:
        token_ids += [tokenizer.pad_token_id] * (max_length - len(token_ids))
        attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
    else:
        token_ids = token_ids[:max_length]
        attention_mask = [1] * max_length
    input_ids = torch.tensor([token_ids])
    attention_mask = torch.tensor([attention_mask])
    embedding = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
    return embedding.tolist()

# step one, tow, three
'''
    create embedding from movies_list (elasticsearch query) 
'''
for item_from_es in movies:
    about = item_from_es['_source']['About']
    tokens = pre_process(about)

    data.append({
        'id':item_from_es['_id'],
        'About':about,
        'embedding': np.array(create_embedding(tokens)[0]),
        
        })

# step four
'''
    createtable and insert data to lance db
'''
tbl = db.create_table("About_table",schema=schema,data=data )
        