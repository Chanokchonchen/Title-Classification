import argparse
from util import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Python Script for Train Title Classification Model')
    parser.add_argument('-input',type=str,help='Path to Train Data in csv format',default=TRAIN_PATH)
    parser.add_argument('-pre',type=bool,help='Use Preprocess File',choices=[True,False],default=False)
    parser.add_argument('-splittest',type=bool, default=False,help="Split Data to evaluate model" ,choices=[True,False])
    parser.add_argument('-engine',type=str,help='Choose Tokenizer Engine',choices=['deepcut','newmm','wordcut'],default='newmm')
    parser.add_argument('-model',type=str,help='Choose Model Support Vector Machine or Random Forest',choices=['svm','rf'],default='svm')
    parser.add_argument('-vectorizer',type=str,help='Chosee Vectorizer',choices=["tfidf","tf"],default="tfidf")
    parser.add_argument('-minterm',type=int,help='Min Term Frequency',choices=range(2, 10),default=4)    
    parser.add_argument('-special',type=str,help='Use Special Features with Method Substring Split or None',choices=["substr","split","None"],default="split")    
    parser.add_argument('-cross',type=bool,help='Use Cross Validation',choices=[True,False],default=False)    
    parser.add_argument('-listdict', default=['wl2'],help='Select Dict for Special Features' , nargs='+',choices=['wl1', 'wl2', 'wl3' , 'wl4'])
    args = parser.parse_args()
    return args

def cross_validate(df,save_name,engine,special_features=False,method='split',model="svm",vector="tfidf",min_count=3,save=False,list_dict=['wl1','wl2','wl3','wl4'],n_folds=5):
  with open(f"{REPORT_PATH}report_{save_name}.txt", "w") as text_file:
      kfold = KFold(5,shuffle=True,random_state=5)
      fold = 1
      max_acc = 0
      acc_list = np.array([])
      pre_list = np.array([])
      re_list = np.array([])
      f1_list = np.array([])
      pn_list = []
      print(f"Model : {save_name}",file=text_file)
      for train, test in kfold.split(df['title_th'].values):
        df_train = df.loc[train].reset_index(drop=True)
        df_test = df.loc[test].reset_index(drop=True)
        y_train = df_train['class'].values
        y_test = df_test['class'].values
        if vector == 'tfidf':
          vectorizer = TfidfVectorizer(tokenizer=tokenizer,min_df=min_count,token_pattern=None)
        else:
          vectorizer = CountVectorizer(tokenizer=tokenizer,min_df=min_count,token_pattern=None)
        sparse_matrix_train = vectorizer.fit_transform(df_train['title_th'].values)
        vector_df_train = pd.DataFrame(sparse_matrix_train.toarray(),columns=vectorizer.get_feature_names_out())
        sparse_matrix_test = vectorizer.transform(df_test['title_th'].values)
        vector_df_test = pd.DataFrame(sparse_matrix_test.toarray(),columns=vectorizer.get_feature_names_out())
        if (special_features):
          if method == "substr":
            X_train = concat_special_features_with_sub_str(df_train,vector_df_train,engine,list_dict).values
            X_test = concat_special_features_with_sub_str(df_test,vector_df_test,engine,list_dict).values
          else:
            X_train = concat_special_features_with_split_title(df_train,vector_df_train,list_dict).values
            X_test = concat_special_features_with_split_title(df_test,vector_df_test,list_dict).values 
        else:
          X_train =  sparse_matrix_train.toarray()
          X_test = sparse_matrix_test.toarray()     
        print(f"Fold {fold} Train features : {X_train.shape[1]} , Test features : {X_test.shape[1]}",file=text_file)
        print(f"Fold {fold} Train features : {X_train.shape[1]} , Test features : {X_test.shape[1]}")
        if model=="svm":
          classifier = SVC(kernel='linear')
        else:
          classifier = RandomForestClassifier(n_estimators=1000, random_state=5)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        re = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        acc_list = np.append(acc_list,acc)
        pre_list = np.append(pre_list,pre)
        re_list = np.append(re_list,re)
        f1_list = np.append(f1_list,f1)
        pn_list.append(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred),file=text_file)
        print(f"Fold {fold} Recall Score : {re}",file=text_file)
        print(f"Fold {fold} Precision Score : {pre}",file=text_file)
        print(f"Fold {fold} F1 Score : {f1}",file=text_file)
        print(f"Fold {fold} Accuracy : {acc}",file=text_file)
        if acc > max_acc:
          max_acc = acc
          if save:
            pickle.dump(vectorizer, open(f"{VECTOR_PATH}{save_name}.pkl", 'wb'))
            pickle.dump(classifier, open(f"{MODEL_PATH}{save_name}.pkl", 'wb'))
            print(f"Fold {fold} Save success Path: {MODEL_PATH}{save_name}.pkl",file=text_file)
            print(f"Fold {fold} Save success Path: {VECTOR_PATH}{save_name}.pkl",file=text_file)
        if fold == n_folds:
          break
        fold+=1
      pn_list = np.array(pn_list).sum(axis=0)
      print(f"All TN FP FN TP : {pn_list.ravel()}",file=text_file)
      if n_folds != 1:
        print(f"Cross validation recall : {np.round(re_list.mean(),4)}",file=text_file)
        print(f"Cross validation precision : {np.round(pre_list.mean(),4)}",file=text_file)
        print(f"Cross validation f1 : {np.round(f1_list.mean(),4)}",file=text_file)
        print(f"Cross validation accuracy : {np.round(acc_list.mean(),4)}",file=text_file)


def train_model(engine,pre_path=False,split_test=False,special_features=False,method='split',model="svm",vector="tfidf",cross=False,min_count=3,list_dict=['wl1','wl2','wl3','wl4']):
      os.makedirs('./Reports', exist_ok=True)
      os.makedirs('./Models', exist_ok=True)
      os.makedirs('./Vectorizers', exist_ok=True)
      save_name = get_name(engine,model,vector,method,special_features,min_count,list_dict)
      print_config(engine,model,vector,method,special_features,min_count,list_dict)
      if cross:
        print(f"Cross Validation : {cross}")
      else:
        print(f"Split Test : {split_test}")
      if pre_path:
        try:
          print(f"Read Preporcess PATH {PRE_PATH}{engine}.csv")
          df = pd.read_csv(f"{PRE_PATH}{engine}.csv",encoding="utf8")
        except:
          sys.exit(f"Preprocess with Engine {engine} Not Found")
      else: 
        print(f"Train Path : {TRAIN_PATH}")
        df = pd.read_csv(TRAIN_PATH,encoding="utf8")
        print(f"Wait for Tokenizing")
        if engine == "wordcut":
          df['title_th'] = df['dc_title_th'].progress_apply(preprocess_wordcut,args=(read_stopwords(STOPWORDS_PATH),DIGIT_SYMBOL))
        else:
          df['title_th'] = df['dc_title_th'].progress_apply(preprocess_title,args=(engine,None,None,None,read_stopwords(STOPWORDS_PATH),DIGIT_SYMBOL)) 
        df = df.drop_duplicates(subset=['title_th']).reset_index(drop=True)
        df.to_csv(f"{PRE_PATH}{engine}.csv",index=False,encoding="utf-8")
        print(f"Save tokenizing result success")
      if cross:
        cross_validate(df,save_name,engine,special_features,method,model,vector,min_count,False,list_dict,5)
      elif split_test:
        cross_validate(df,save_name,engine,special_features,method,model,vector,min_count,True,list_dict,1)
      else:
        vectors = process_vector(df['title_th'].values,save_name,min_count,vector)
        if special_features:
          if method=='substr':
            vectors = concat_special_features_with_sub_str(df,vectors,engine,list_dict)
          else:
            vectors = concat_special_features_with_split_title(df,vectors,list_dict)
        X = vectors.values
        y = df['class'].values
        print(f"X shape {X.shape} , Y shape {y.shape}")
        if model=="svm":
          classifier = SVC(kernel='linear')
        else:
          classifier = RandomForestClassifier(n_estimators=1000, random_state=5)
        print(f'Model : {model}')
        classifier.fit(X, y)
        pickle.dump(classifier, open(f"{MODEL_PATH}{save_name}.pkl", 'wb'))
        print(f"Save success Path: {MODEL_PATH}{save_name}.pkl")

args = parse_arguments()
engine = args.engine
list_dict = args.listdict
pre = args.pre
model = args.model
vector = args.vectorizer
special = False
method = "split"
min_count = args.minterm
split_test = args.splittest
cross = args.cross
TRAIN_PATH = args.input
if (args.special != "None"):
  special = True
  method = args.special

train_model(engine=engine,pre_path=pre,split_test=split_test,special_features=special,method=method,model=model,vector=vector,cross=cross,min_count=min_count,list_dict=list_dict)
