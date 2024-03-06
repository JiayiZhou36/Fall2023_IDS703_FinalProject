# Fall2023_IDS703_FinalProject
This is for class NLP YouTube category classification.

### Introduction
The goal of this project is to classify the YouTube videos based on the tags of videos (a form of the descriptions for videos that enables users to search for content). It automates the categorization process so that it can improve the efficiency of categorization and potentially help advertisements targeting.

The data we’re using is from [Kaggle](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset?select=US_youtube_trending_data.csv). It contains data from multiple countries, and we only selected the video records from the United States. Each row represents the tags for one specific video. One Video only belongs to one category. We’re specifically interested in whether the video belongs to music or sports category, which are the two major categories in the data. Each category has around 30,000 records.

The original dataset includes 16 columns and almost 30 categories. We only include the music and sports category. We removed the characters that are not in English characters or numbers to exclude other languages. We also converted the characters into lowercase. We tokenize our data into the form of lists of lists. Each record is a list including word as elements, and the lists of each record concatenate together as one large list. Finally, we stored the tokens of the two categories into separate txt files.

### Methods
#### Generative Probablistic Model--Naive Bayes Model
We used Naive Bayes model as our generative probabilistic model for this classification problem.

The formula for Naive Bayes algorithm is argmax(P(d|c)P(c)). The logic behind the model is to maximize the probability of each tag represented in each category. The assumption for this model is that the sequence doesn’t matter and feature probabilities P(d|c) are independent given the class. To avoid extremely small probability, we are adding up log probabilities for each word in each category and the probability of each category.

In our model, first, the words are converted into embeddings using raw count and TF-IDF. Then, we train the data by MultinomialNB().

The accuracy is calculated based on the percentage of data correctly categorized among all data. Besides, we include a timer to track the time it spends on the training process.

#### Discrimitive Neural Network--RNN
For the discriminative neural network, we designed a fundamental recurrent neural network model tailored for text classification, specifically aimed at categorizing YouTube videos into two categories based on their tags.

In this neural network architecture, the process begins by converting input words into dense vectors through an embedding layer. The nn.Embedding(input_size, hidden_size) operation facilitates the mapping of words (represented as indices) to dense vectors of a predetermined size (hidden_size). Subsequently, the embedded sequences undergo processing by a recurrent layer. The nn.RNN(hidden_size, hidden_size, batch_first=True) layer handles input sequences of embedded vectors, generating output sequences. This layer employs a state of size hidden_size, initialized to 64. Following this, the output from the RNN layer is channeled through a linear layer. The nn.Linear(hidden_size, output_size) operation maps the RNN output to the final output classes, which are binary—specifically, music or sports.

To optimize the model's performance, the output is juxtaposed with the actual labels using the cross-entropy loss. The Adam optimizer is then enlisted to minimize this loss. The training loop embarks on a series of epochs, with the current run comprising 20. Within each epoch, the DataLoader (dataloader) facilitates the iteration over batches of data. To ensure effective parameter updates, the optimizer's gradients are reset to zero using optimizer.zero_grad(). The model's output for the input batch is computed (outputs = model(inputs)), and subsequently, the loss is calculated, and backpropagation is executed (loss.backward()). The optimizer then takes a step, updating the model parameters based on the computed gradients (optimizer.step()).

Throughout the training process, key metrics such as total loss, correct predictions, and total samples are dynamically updated. Accuracy is computed based on the ratio of correct predictions to the total number of samples. After each epoch, the average loss and accuracy for that specific epoch are printed. The overall training time is determined by recording the start and end times of the training process.
