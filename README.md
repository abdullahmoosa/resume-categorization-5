# Resume Categorization

### Steps:
  1. If possible create a virtual environment in python.
  2. Then inside the virtual environment clone the github repo.
  3. Use the requirement.txt to install the dependencies.
  4. **PLEASE download the model** : https://mega.nz/file/Eq0jATbJ#LEmoVJzASIgJ_T88UjRAO9q9H1QK7DzxhPYYYwkWtWA
  5. Put the model in the same directory as scripts.py **(Makse sure the name of model is "bert_model.h5")**
  6. I was not able to upload the model due to its huge size.
  7. Run script.py from the command line as intended : python script.py "directory". Make sure you are in the same directory as script.py
  8. resume-categorization (2).ipynb contains the model training and documentation guide.


### Important Findings:
  1. BERT performs the best.
  2. The dataset is imbalanced. Therefore the accuracy is not good for some classes.
  3. **For further details please review "resume-categorization (2).ipynb" .**
     
