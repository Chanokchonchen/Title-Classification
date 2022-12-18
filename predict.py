import argparse
from util import *


def parse_arguments():
    model_names = get_allmodelname()
    parser = argparse.ArgumentParser(description='Python Script for Predict Title Category')
    parser.add_argument('-input',type=str,help='Path to Prediction Data in csv format',default=TEST_PATH)
    parser.add_argument('-output',type=str,help='Path to Prediction Result in csv format',default=OUTPUT_PATH)
    parser.add_argument('-modelname',type=str,help='Choose Model Name',choices=model_names,required=True)  
    args = parser.parse_args()
    return args

def prediction(input_path,output_path,model_name):
    date , engine , special_features , method , model , vector , min_count , list_dict = parse_modelname(modelname=model_name)
    load_name = get_name(date,engine,model,vector,method,special_features,min_count,list_dict)
    print_config(date,engine,model,vector,method,special_features,min_count,list_dict)
    test_df = pd.read_csv(input_path,encoding = 'utf8')
    print(f"Input Path : {input_path}")
    print(f"Output Path : {output_path}\n")
    if engine=="wordcut":
      test_df['title_th'] = test_df['dc_title_th'].progress_apply(preprocess_wordcut,args=(read_stopwords(STOPWORDS_PATH),DIGIT_SYMBOL))
    else:
      test_df['title_th'] = test_df['dc_title_th'].progress_apply(preprocess_title,args=(engine,None,None,None,read_stopwords(STOPWORDS_PATH),DIGIT_SYMBOL))
    try:
      print(f"Vectorizer Path : {VECTOR_PATH}{load_name}.pkl")
      with open(f"{VECTOR_PATH}{load_name}.pkl", 'rb')as f:
        vectorizer = pickle.load(f)
    except:
      sys.exit("Vectorizer with this config can't be found")
    test_vectors = sparse_matrix_to_data_frame(vectorizer.transform(test_df['title_th'].values),vectorizer=vectorizer)
    if special_features:
      if method=='substr':
        print(f"Special Method : Split title & dict")
        test_vectors = concat_special_features_with_sub_str(test_df,test_vectors,engine,list_dict)
      else:
        print(f"Special Method : Split title only")
        test_vectors = concat_special_features_with_split_title(test_df,test_vectors,list_dict)
    print(f"Number of title : {test_vectors.shape[0]}")
    print(f"Number of features : {test_vectors.shape[1]}")
    try:
      print(f"Model Path : {MODEL_PATH}{load_name}.pkl")
      with open(f"{MODEL_PATH}{load_name}.pkl", 'rb')as f:
        model = pickle.load(f)
    except:
      sys.exit("Model with this config can't be found")
    print("Wait for Predict Result")
    prediction_results = model.predict(test_vectors.values)
    test_df.drop(['title_th'],axis=1,inplace=True)
    print("Predict Success")
    return pd.concat([test_df,pd.DataFrame(data={'Predict':prediction_results})],axis=1).to_csv(output_path,index=False,encoding="utf-8")

args = parse_arguments()
model_name = args.modelname
TEST_PATH = args.input
OUTPUT_PATH = args.output

prediction(TEST_PATH,OUTPUT_PATH,model_name=model_name)