import re
import os
import sys
import pickle
import subprocess
import numpy as np
import pandas as pd
from config import *
from typing import List
from tqdm import tqdm
from pythainlp.tokenize import word_tokenize as pythainlp_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split ,  cross_val_score , KFold
from sklearn.metrics import classification_report , accuracy_score , precision_score , recall_score , f1_score , confusion_matrix
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from marisa_trie import Trie
from datetime import datetime
tqdm.pandas()

DIGIT_SYMBOL = '[:NUM:]'
WHITE_SPACE = ' '
SPECIAL_SYMBOL = r'["' + r"!#$%&'()*+,-./:;<=>?@[\]^_`{|}~" + r'“”‘ๆฯ–]+'
SPECIAL_FEATURE = 4

class Text:
    def __init__(self,
                 value: str,
                 ignore_tokenization: bool):
        self.value = value
        self.ignore_tokenization = ignore_tokenization

    def __repr__(self) -> str:
        return str(self.value)

def tokenize(text: str,
             custom_dict : List[str],
             tokenizer_engine: str,
             grouping_words: List[str]) -> List[Text]:
    tokenize_result = []
    for text in split_by_grouping_words(text=text,
                                        grouping_words=grouping_words):
        if text.ignore_tokenization:
            tokenize_result.append(text)
        else:
            for word in pythainlp_tokenize(text.value,custom_dict=custom_dict,
                                           engine=tokenizer_engine):
                if word != WHITE_SPACE:
                    tokenize_result.append(
                        Text(word, ignore_tokenization=False))
    return tokenize_result

def preprocess_title(title: str,
                        tokenizer_engine: str,
                        grouping_words: List[str],
                        normalization_words: dict,
                        custom_dict : List[str],
                        stopwords: List[str],
                        digitsymbol: str) -> str:
    if custom_dict:
      custom_dict = Trie(custom_dict)
    tokenized_words = tokenize(text=title,
                               custom_dict=custom_dict,
                               tokenizer_engine=tokenizer_engine,
                               grouping_words=grouping_words)
    if normalization_words:
      normalized_words = list(
          map(lambda word:
              normalize(text=word,
                        normalization_words=normalization_words),
              tokenized_words))
    else:
      normalized_words = tokenized_words
    removed_stopword_words = remove_stopword(texts=normalized_words,
                                             stopwords=stopwords)
    space_separated_words = re.sub(r'\n+','',re.sub(r'\d+', digitsymbol, re.sub(SPECIAL_SYMBOL,''," ".join(removed_stopword_words).lower()))).replace('\\','')
    final_result = " ".join(re.findall(r'\S+', space_separated_words))
    # print(final_result)
    return final_result

def split_by_grouping_words(text: str,
                            grouping_words: List[str]) -> List[Text]:

    before_split_by_grouping_words_texts = [Text(value=text, ignore_tokenization=False)]
    after_split_by_grouping_words_texts = []
    if grouping_words:
      for grouping_word in grouping_words:
          for before_tokenize_text in before_split_by_grouping_words_texts:
              for word in split_by_grouping_word(
                      text=before_tokenize_text,
                      grouping_word=grouping_word):
                  after_split_by_grouping_words_texts.append(word)
          before_split_by_grouping_words_texts = after_split_by_grouping_words_texts
          after_split_by_grouping_words_texts = []   
    splitted_by_grouping_words_texts = before_split_by_grouping_words_texts
    return splitted_by_grouping_words_texts

def split_by_grouping_word(text: Text, grouping_word: str):
    if text.ignore_tokenization:
        return [text]

    split_results = re.split(f'({grouping_word})', text.value)
    return list(map(
        lambda split_result:
        Text(value=split_result,
             ignore_tokenization=split_result == grouping_word),
        split_results))
  
def normalize(text: Text, normalization_words: dict) -> Text:
    if text.ignore_tokenization:
        return text

    # result = deepcopy(text)
    result = text

    for original_word in normalization_words:
        new_word = normalization_words[original_word]
        if result.value == original_word:
          result.value = new_word
    return result

def remove_stopword(texts: List[Text], stopwords: List[str]) -> List[str]:
    return [text.value for text in texts if (text.value not in stopwords) or text.ignore_tokenization]
  

def tokenizer(x: str):
    l = re.split(r'\s+',x)
    p = re.compile('[\S+]{'+str(MIN_LENGTH)+',}')
    return [i for i in l if p.match(i)]
  
    
def sparse_matrix_to_data_frame(sparse_matrix, vectorizer):
    doc_term_matrix = sparse_matrix.toarray()
    df = pd.DataFrame(doc_term_matrix,
                      columns=vectorizer.get_feature_names_out (),
                      )
    return df

def read_stopwords(filename):
    print(f"Stopwords Path {filename}")
    def process_line(line: str):
        return line.strip()

    with open(filename) as f:
        lines = f.readlines()

    processed_lines = list(map(process_line, lines))

    stop_words = processed_lines

    return stop_words  

def read_dict(filename):
    print(f"Dict Path {filename}")
    def process_line(line: str):
        return line.strip()

    with open(filename) as f:
        lines = f.readlines()

    processed_lines = list(map(process_line, lines))
    return processed_lines



def get_dict_features_split_title(title,wordlist):
    for word in title.split(" "):
      if word in wordlist:
        # print(f"{word} {title}")
        return 1
    return 0

def get_dict_features_sub_str(title,wordlist):
    for word in wordlist:
      if pad_spaces(word) in pad_spaces(title):
        # print(f"{pad_spaces(word)} {pad_spaces(title)}")
        return 1
    return 0

def preprocess_wordcut(text,stop_words,digitsymbol):
    pipe = subprocess.Popen(["bash","wordcut.sh",f"'{text}'"],stdout=subprocess.PIPE,encoding='utf8')
    before_result = pipe.communicate()[0]
    tokenize_result = re.split(r'\s+',before_result)
    removed_stopword_words  = [word for word in tokenize_result if(word not in stop_words)]
    space_separated_words = re.sub(r'\n+','',re.sub(r'\d+',digitsymbol, re.sub(SPECIAL_SYMBOL,''," ".join(removed_stopword_words).lower()))).replace('\\','')
    final_result = " ".join(re.findall(r'\S+', space_separated_words))
    return final_result

def clean(text):
  res = " ".join(re.findall(r'\S+', text))
  if (text != res):
    print(f"Bef {text}")
    print(f"Aft {res}")
  return res

def cleandf(path):
    print(f"Datasets Path {path}")
    df = pd.read_csv(path,encoding = 'utf8')  
    df['title_th'] = df['title_th'].progress_apply(clean)
    df.to_csv(path,index=False,encoding="utf-8")  

def create_wordlist(wl):
    for root, dirs, files in os.walk(f"{DICT_PATH}{wl}", topdown=False):
      final_set = set()
      for name in files:
          final_set = final_set.union(set(read_dict(os.path.join(root, name))))
      sort_final_list = list(sorted(final_set))
    np.savetxt(f'{DICT_PATH}{wl}.txt', sort_final_list, delimiter="\n", fmt="%s")

def tokenize_title(engine,path,with_dict=False,min_l=0):
  df = pd.read_csv(f"{path}input.csv",encoding="utf8")
  stop_words = read_stopwords(STOPWORDS_PATH)
  trie = None
  if with_dict:
    plant_dict = set(read_dict(f"{DICT_PATH}Plant_dict.txt"))
    micro_dict = set(read_dict(f"{DICT_PATH}Microorganism_dict.txt"))
    animal_dict = set(read_dict(f"{DICT_PATH}Animal_dict.txt"))
    all_dict = list(plant_dict.union(animal_dict.union(micro_dict)))
    pre_dict = [word for word in all_dict if len(word)>min_l]
    trie = Trie(pre_dict)
    save_path = f"{path}{engine}_dict_{min_l}.csv"
    print(f"Engine {engine} Minimum length word in dict {min_l} Word Count {len(pre_dict)}")
  else:
    save_path = f"{path}{engine}.csv"
  print(f"Save path : {save_path}")
  print(f"Tokenizing ...")
  if engine == 'wordcut':
    df['title_th'] = df['dc_title_th'].progress_apply(preprocess_wordcut,args=(stop_words,))
  else:
    df['title_th'] = df['dc_title_th'].progress_apply(preprocess_title,args=(engine,None,None,trie,stop_words,DIGIT_SYMBOL))
  df.to_csv(save_path,index=False,encoding="utf-8") 
  df = df[df['title_th'] != ""].reset_index(drop=True)
  print(f"Save path : {save_path}")
  df.to_csv(save_path,index=False,encoding="utf-8")  

def tokenize_dict(engine,d_name):
  l_dict = read_dict(f"{DICT_PATH}{d_name}.txt")
  print(f"Read Dict Path {DICT_PATH}{d_name}.txt")
  stop_words = read_stopwords(STOPWORDS_PATH)
  if engine=='wordcut':
      result =   list(map(lambda word: preprocess_wordcut(word,stop_words,DIGIT_SYMBOL),l_dict))
  else:
      result =  list(map(lambda word: preprocess_title(word,engine,None,None,None,stop_words,DIGIT_SYMBOL),l_dict))
  np.savetxt(f'{DICT_PATH}{d_name}_{engine}.txt', result, delimiter="\n", fmt="%s")
  print(f"Save Success Path {DICT_PATH}{d_name}_{engine}.txt")

def pad_spaces(text):
  return " " + text + " "

def read_preprocess(engine,with_dict=False,min_l=0):
  if with_dict:
    print(f"Datasets Path {PRE_PATH}{engine}_dict_{min_l}.csv")
    return pd.read_csv(f"{PRE_PATH}{engine}_dict_{min_l}.csv",encoding = 'utf8')
  else:
    print(f"Datasets Path {PRE_PATH}{engine}.csv")
    return pd.read_csv(f"{PRE_PATH}{engine}.csv",encoding = 'utf8')

def process_vector(data,min_count=3,vector="tfidf"):
    if vector=="tfidf":
      vectorizer = TfidfVectorizer(tokenizer=tokenizer,min_df=min_count,token_pattern=None)
      print(f"Vectorizer TFIDF Min Token Length {MIN_LENGTH} Min Count {min_count}")
    else:
      vectorizer = CountVectorizer(tokenizer=tokenizer,min_df=min_count,token_pattern=None)
      print(f"Vectorizer TF Min Token Length {MIN_LENGTH} Min Count {min_count}")
    sparse_matrix = vectorizer.fit_transform(data)
    return sparse_matrix_to_data_frame(sparse_matrix=sparse_matrix,vectorizer=vectorizer) , vectorizer

def concat_special_features_with_split_title(df,vectors,list_dict=["wl1","wl2","wl3","wl4"]):
    wl_features = []
    for d in list_dict:
      wl = set(read_dict(f"{DICT_PATH}{d}.txt"))
      wl_features.append(df['title_th'].apply(get_dict_features_split_title,args=(wl,)).values)
    dict_df = pd.DataFrame(columns=list_dict,data=np.array(wl_features).reshape(-1,len(list_dict)))
    print(f"Match Documents : {dict_df.sum().sum()}")
    return pd.concat([vectors,dict_df],axis=1)

def concat_special_features_with_sub_str(df,vectors,engine,list_dict=["wl1","wl2","wl3","wl4"]):
    wl_features = []
    try:
      for d in list_dict:
        wl = set(read_dict(f"{DICT_PATH}{d}_{engine}.txt"))
        wl_features.append(df['title_th'].apply(get_dict_features_sub_str,args=(wl,)).values)
    except:
      sys.exit(f"Dict files with {engine} not found. Please run command 'python3 tokenize_dict.py -engine={engine}'")
    dict_df = pd.DataFrame(columns=list_dict,data=np.array(wl_features).reshape(-1,len(list_dict)))
    print(f"Match Documents : {dict_df.sum().sum()}")
    return pd.concat([vectors,dict_df],axis=1)

def drop_duplicate(df):
    return df.drop_duplicates(subset=['title_th'])

def get_allmodelname():
    model_lists = []
    for root, dirs, files in os.walk(f"{MODEL_PATH}", topdown=False):
      for name in files:
        if name.endswith('.pkl'):
          model_lists.append(name[:-4])
    model_lists.sort(reverse=True)
    return model_lists

def parse_modelname(modelname):
  date = "_".join(modelname.split('_')[:6])
  split_name = modelname.split('_')[6:]
  engine = split_name[0]
  model = split_name[1]
  vector = split_name[2]
  if len(split_name) <= 4:
    minterm = split_name[3]
    special_features = False
    method = "split"
    list_dict =['wl1', 'wl2', 'wl3' , 'wl4']
  else:
    special_features = True
    method = split_name[3]
    minterm = split_name[4]
    dicts = split_name[5]
    list_dict = [dicts[index.start():index.end()] for index in re.finditer(r"wl\d+",dicts)]
  return date , engine , special_features , method , model , vector , int(minterm) , list_dict


def print_config(date,engine,model,vector,method,special,min_count,listdict):
  print("===========================================")
  print(f"Tokenizer Engine : {engine}")
  print(f"Model : {model}")
  print(f"Vectorizer method : {vector}")
  print(f"Min Count : {min_count}")
  print(f"Date Time : {datetime.strptime(date, '%y_%m_%d_%H_%M_%S').strftime('%y/%m/%d %H:%M:%S')}")
  if special:
    print(f"Special Feature Method : {method}")
    print(f"List Dicts : {listdict}")
  print("===========================================\n")

def get_name(date,engine,model,vector,method,special,min_count,listdict):
  if special:
    d = ''.join(listdict)
    return f"{date}_{engine}_{model}_{vector}_{method}_{min_count}_{d}"
  return f"{date}_{engine}_{model}_{vector}_{min_count}"

