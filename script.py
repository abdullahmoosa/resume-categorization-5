# These import statements are importing various libraries and modules that are required for the code

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import re
from transformers import BertTokenizer, TFBertForSequenceClassification
import fitz  # PyMuPDF
import shutil
from tensorflow.keras.models import load_model

import sys


# Main directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def analyze_directory(directory_path):

    """_summary_
    "analyze_directory" is used to check if the directory that is written in the command line is valir or not. If it is valid then it converts the pdfs of the directory to texts and then returns it along with the name of each file.

    Returns:
        list containing converted text of each pdf and list containing the name of each pdf
        
    """

    all_pdfs = []
    all_pdf_names = []
    try:
        # Check if the path is a valid directory
        if not os.path.isdir(directory_path):
            print("Error: The provided path is not a valid directory.")
            return

        # List the contents of the directory
        contents = os.listdir(directory_path)
        
        # Print the contents
        # print(f"Contents of directory '{directory_path}':")
        
        for item in contents:
            item_path = os.path.join(directory_path, item)
            if item.lower().endswith('.pdf') and os.path.isfile(item_path):
                # Convert PDF to text
                pdf_text = pdf_to_text(item_path)
                all_pdfs.append(pdf_text)
                all_pdf_names.append(item)
                
        
        return all_pdfs, all_pdf_names
    except Exception as e:
        print(f"An error occurred: {e}")


def pdf_to_text(pdf_path):
    """
    The function `pdf_to_text` takes a path to a PDF file as input and returns the text content of the
    PDF file.
    
    :param pdf_path: The `pdf_path` parameter is the file path to the PDF document that you want to
    convert to text
    :return: the text extracted from the PDF file.
    """
    text = ""
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text("text")
    
    pdf_document.close()
    return text

# The ML model returns a integer representing the class. So we use a dictionary to get the actual value from the ML model output
category_labels = {
    0: 'HR',
    1: 'DESIGNER',
    2: 'INFORMATION-TECHNOLOGY',
    3: 'TEACHER',
    4: 'ADVOCATE',
    5: 'BUSINESS-DEVELOPMENT',
    6: 'HEALTHCARE',
    7: 'FITNESS',
    8: 'AGRICULTURE',
    9: 'BPO',
    10: 'SALES',
    11: 'CONSULTANT',
    12: 'DIGITAL-MEDIA',
    13: 'AUTOMOBILE',
    14: 'CHEF',
    15: 'FINANCE',
    16: 'APPAREL',
    17: 'ENGINEERING',
    18: 'ACCOUNTANT',
    19: 'CONSTRUCTION',
    20: 'PUBLIC-RELATIONS',
    21: 'BANKING',
    22: 'ARTS',
    23: 'AVIATION'
}


def remove_punctuation(text):
    """
    The function `remove_punctuation` takes a text as input and removes all punctuation marks from it,
    replacing them with spaces.
    
    :param text: The input text from which you want to remove punctuation marks
    :return: the input text with all punctuation marks removed.
    """
    punctuations = '''“”!()-[]{};:'"\,<>./?@#$%^&*_~�।’‘Ôø√ºß√√≥'''
    review = text.replace('\n', ' ')
    no_punct = ""
    for char in review:
        if char not in punctuations:
          no_punct = no_punct + char
        else:
          no_punct += " "

    return no_punct

def text_preprocess(text):
    """
    The function `text_preprocess` takes in a string of text, removes URLs, numbers, and punctuation,
    converts all words to lowercase, and returns the processed text as a single string.
    
    :param text: The input text that needs to be preprocessed
    :return: a preprocessed version of the input text.
    """

    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[0-9]', ' ', text)
    text = remove_punctuation(text)
    words = text.split()
    process_words = []
    for word in words:
        if re.match(r'^[a-zA-Z]+$', word):
            process_words.append(word.lower())
        else:
            process_words.append(word)
    
    return ' '.join(process_words)

def process_and_extract_informations(unprocessed_resumes):
    """
    The function takes a list of unprocessed resumes, processes each resume using a text_preprocess
    function, and returns a list of processed resumes.
    
    :param unprocessed_resumes: A list of unprocessed resumes, where each resume is a string
    :return: a list of processed resumes.
    """
    processed_resumes = [text_preprocess(text) for text in unprocessed_resumes]
    return processed_resumes
    

def custom_objects_scope():
    # Necessary for loading custom keras model.
    return {'TFBertForSequenceClassification': TFBertForSequenceClassification}

# # Load the Keras model with the custom object scope

def load_bert_model():
    """
    The function `load_bert_model` loads a BERT model from a specified file path.
    :return: The function `load_bert_model` returns the loaded BERT model.
    """
    model_path = os.path.join(script_dir, 'bert_model.h5')
    loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects_scope())
    return loaded_model


def return_process_inputs_bert(input_text):
    """
    The function `return_process_inputs_bert` takes an input text and returns processed inputs using the
    BERT tokenizer.
    
    :param input_text: The input text that you want to process using BERT
    :return: the processed inputs for the BERT model.
    """
    inputs = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    max_length=100,
    padding='max_length',
    truncation=True,
    return_tensors='tf'
    )
    return inputs

def return_category(class_no):
    # returns category name from ID
    return category_labels[class_no]

def make_prediction(model,inputs):
    """
    The function `make_prediction` takes a model and inputs, and returns the predicted class name based
    on the model's prediction.
    
    :param model: The model parameter refers to a trained machine learning model that is capable of
    making predictions. 
    :param inputs: The `inputs` parameter is a dictionary that contains the input data for the model. It
    has two keys: 'input_ids' and 'attention_mask'
    :return: the predicted class name.
    """
    outputs = model.predict([inputs['input_ids'], inputs['attention_mask']])
    predicted_class = outputs.argmax()
    predicted_class_name = return_category(predicted_class)
    return predicted_class_name

all_output_directories = []
all_outout_categories = []
def add_to_output_directory(pdf_name,output):
    """
    The function `add_to_output_directory` moves or copies a PDF file to a specified output directory.
    
    :param pdf_name: The name of the PDF file that you want to add to the output directory
    :param output: The "output" parameter is the name of the subdirectory within the "prediction"
    directory where the PDF file will be moved or copied to
    :return: nothing if the destination path already exists as a file.
    """

    output_dir = os.path.join(script_dir, output)
    all_output_directories.append(output_dir)
    destination_path = os.path.join(output_dir, pdf_name)
    item_path = os.path.join(directory_path, pdf_name)
    if os.path.exists(output_dir):
        destination_path = os.path.join(output_dir, pdf_name)
        if os.path.isfile(destination_path):
            return
        shutil.move(item_path, destination_path)
    else:
        os.mkdir(output_dir)
        destination_path = os.path.join(output_dir, pdf_name)
        if os.path.isfile(destination_path):
            return
        shutil.copy(item_path, destination_path)

def predict_all_pdfs(resumes,model,pdf_names):
    """
    The function `predict_all_pdfs` takes a list of resumes, a model, and a list of PDF names as input,
    processes each resume using a function called `return_process_inputs_bert`, makes a prediction using
    the provided model, and adds the output to an output directory along with the corresponding PDF
    name.
    
    :param resumes: A list of resumes in PDF format that you want to make predictions on
    :param model: The "model" parameter refers to the machine learning model that will be used for
    making predictions on the resumes. 
    :param pdf_names: pdf_names is a list of names of the PDF files
    """


    for index,resume in enumerate(resumes):
        inputs = return_process_inputs_bert(resume)
        output = make_prediction(model=model, inputs=inputs)
        all_outout_categories.append(output)
        add_to_output_directory(pdf_name= pdf_names[index],output=output)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please enter relative directory correctly")
    else:
        directory_path = sys.argv[1]
        all_pdfs, all_pdf_names = analyze_directory(directory_path)
        processed_resumes = process_and_extract_informations(all_pdfs)

        model_name = 'bert-base-uncased'  
        tokenizer = BertTokenizer.from_pretrained(model_name)

        model = load_bert_model()

        predict_all_pdfs(processed_resumes,model, all_pdf_names)
        data = {'filename': all_output_directories, 'categories': all_outout_categories}
        category_df = pd.DataFrame(data)

        category_df.to_csv('categorized_resumes.csv', index= False)


