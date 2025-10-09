## 1.5 Model Selection Process

<a href="https://www.youtube.com/watch?v=OH_R0Sl9neM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=6"><img src="images/thumbnail-1-05.jpg"></a>

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-15-model-selection-process)


## Notes

### Which model to choose?

- Logistic regression
- Decision tree
- Neural Network
- Or many others

The validation dataset is not used in training. There are feature matrices and y vectors
for both training and validation datasets. 
The model is fitted with training data, and it is used to predict the y values of the validation
feature matrix. Then, the predicted y values (probabilities)
are compared with the actual y values. 

**Multiple comparisons problem (MCP):** just by chance one model can be lucky and obtain
good predictions because all of them are probabilistic. 

The test set can help to avoid the MCP. Obtaining the best model is done with the training and validation datasets, while the test dataset is used for assuring that the proposed best model is the best. 

1. Split datasets in training, validation, and test. E.g. 60%, 20% and 20% respectively 
2. Train the models
3. Evaluate the models
4. Select the best model 
5. Apply the best model to the test dataset 
6. Compare the performance metrics of validation and test

<u>NB:</u> Note that it is possible to reuse the validation data. After selecting the best model (step 4), the validation and training datasets can be combined to form a single training dataset for the chosen model before testing it on the test set.





### the 6 steps

1. Split datasets (60%-20%-20%)
2. Train the model
3. Apply the model to validation dataset
   Repeat 2 and 3 a few times
4. Select the best model
5. Apply the model to the test dataset
6. Check everything is good (compare accuracy of validation and test datasets)

### Alternative Approach

To not waste the validation dataset you can reuse it. That means you  train a model on the training dataset, apply the model on validation  dataset, and choose the best model as before. But then combine train and validation datasets and train another model based on both datasets.  Apply this new model on the test dataset.

The alternative approach mentioned above, where the validation  dataset is not wasted, can be a practical solution in some cases. By  combining the training and validation datasets, we can create a larger  dataset for training a new model. This approach can help improve the  performance and generalization of the selected model.

Here are the steps for the alternative approach:

1. Split the original dataset into training, validation, and test sets with a ratio of 60%-20%-20%.
2. Train the initial models using the training dataset.
3. Apply the initial models to the validation dataset and evaluate their performance.
4. Select the best-performing model based on the validation results.
5. Combine the training and validation datasets to create a new combined dataset.
6. Retrain the selected model using the new combined dataset.
7. Apply the newly trained model to the test dataset to assess its performance on unseen data.

By training the model on a larger combined dataset, we can  potentially capture more patterns and improve the model’s ability to  generalize to new data. The final evaluation on the test dataset  provides a more reliable measure of the model’s performance and gives us confidence in its ability to make accurate predictions.

It’s important to note that the alternative approach may not always  yield better results compared to the original model selection process.  The effectiveness of this approach depends on the specific  characteristics of the dataset and the performance of the initial  models. Experimentation and careful evaluation are key to determine the  most suitable approach for your machine learning task.

In summary, the model selection process is a crucial step in machine  learning, and it involves thoroughly assessing different models and  selecting the one that performs the best on unseen data. The alternative approach of combining the training and validation datasets can be an  effective strategy to enhance model performance and generalize better.





<table>
   <tr>
      <td>⚠️</td>
      <td>
         The notes are written by the community. <br>
         If you see an error here, please create a PR with a fix.
      </td>
   </tr>
</table>

* [Notes from Peter Ernicke](https://knowmledge.com/2023/09/13/ml-zoomcamp-2023-introduction-to-machine-learning-part-5/)

## Navigation

* [Machine Learning Zoomcamp course](../)
* [Lesson 1: Introduction to Machine Learning](./)
* Previous: [CRISP-DM](04-crisp-dm.md)
* Next: [Setting up the Environment](06-environment.md)
