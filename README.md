# Resume Categorization

This is a script which can categorize resumes into 24 different classes. 

The containing classes are : HR, DESIGNER, INFORMATION-TECHNOLOGY, TEACHER, ADVOCATE, BUSINESS-DEVELOPMENT, HEALTHCARE, FITNESS, AGRICULTURE, BPO, SALES, CONSULTANT, DIGITAL-MEDIA, AUTOMOBILE, CHEF, FINANCE, APPAREL, ENGINEERING, ACCOUNTANT, CONSTRUCTION, PUBLIC-RELATIONS, BANKING, ARTS, AVIATION. 

The output will be saved into a different directory named "prediction" . Inside the "prediction" folder, the resumes will be categorized inside the cateogy of each resume.

**Dataset used :** https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

**Example output folder structure:**

```
prediction/
│
├── HR/
│ ├── hr_resume1.pdf
│ ├── hr_resume2.pdf
│ └── ...
│
├── ACCOUNTANT/
│ ├── accountant_resume1.pdf
│ ├── accountant_resume2.pdf
│ └── ...
│
├── TEACHER/
│ ├── teacher_resume1.pdf
│ ├── teacher_resume2.pdf
│ └── ...
│
└── ...
```

### Steps to Run:
  1. Create a virtual envitonment in python.
  2. Then inside the virtual environment clone the github repo.
  3. Use the requirement.txt to install the dependencies.
  4. **PLEASE download the model** : https://mega.nz/file/Eq0jATbJ#LEmoVJzASIgJ_T88UjRAO9q9H1QK7DzxhPYYYwkWtWA
  5. Put the model in the same directory as scripts.py **(Makse sure the name of model is "bert_model.h5")**
  6. Run script.py from the command line as intended : python script.py "directory". Make sure you are in the same directory as script.py
  7. resume-categorization (2).ipynb contains the model training and documentation guide.

#### Creating Virtual Environment:
Go to the directory you want to clone the repo. 
Open command line on that directory.

  ``` command-line
pip install venv
python -m venv "name of virtual environment"
git clone https://github.com/abdullahmoosa/resume-categorization-final.git
cd resume-catogirization-final
pip install -r requirements.txt
```

#### Running the script:
After installing requirement.txt and putting the bert_model.h5 in the same directory as the script.py, 
```
python script.py path_to_directory_containing_the_resume_pdfs
```

Here replace 'path_to_directory_containing_the_resume_pdfs' with the actual directory containing the pdfs.

### Model Creation and Steps: 
  1. First preprocess the texts - remove punctuations, remove stopwords etc.
  2. Tokenize the inputs.
  3. Generate word vectors.
  4. Train various models like - CNN,LSTM,BERT on the input data and evaluate the accuracy.

### Important Findings:
  1. BERT performs the best.
  2. The dataset is imbalanced. Therefore the accuracy is not good for some classes.
  3. **For further details please review "resume-categorization (2).ipynb" .**

**Correct Prediction of Model per class :**

![download (2)](https://github.com/abdullahmoosa/resume-categorization-final/assets/67234038/b980e0d3-d960-46ca-9cd0-5dd4ae9e8731)


