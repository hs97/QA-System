# QA-System

To run the QA System, cd into hw5_data
The main file is contained in qasystem.py

Currently, the code is set to run the test questions. To switch to the training questions, replace 'topdocs/test/top_docs.' with 'topdocs/train/top_docs.' in the passage_retrieval() method and set the train_qPATH on the bottom of the code to 'qadata/train/questions.txt'

The output that is printed to terminal can be saved in a text file, then compared with the provided answer patterns using evaluation.py.