# MSCS_634_Lab_2

The purpose of this lab was to get hands-on experience with KNN and RNN classification algorithms.

When I was performing the initial data exploration, such as df.describe, target count, and null value count in one Python block, I was surprised that only the results of the last line of code were displayed. After some Googling, I found out that it is a feature of Jupyter Notebooks. In order to display other lines as well, I had to wrap the other lines in display()

I split the data into training and test parts in the ratio of 8:2. I then applied the KNN algorithm with k values from the assignment. I was able to achieve the following accuracies:
- k = 1 --> Accuracy: 0.7778
- k = 5 --> Accuracy: 0.7222
- k = 11 --> Accuracy: 0.7500
- k = 15 --> Accuracy: 0.7500
- k = 21 --> Accuracy: 0.7778

An interesting insight I can point out is that I got the best accuracy with k=1 and k=21, even though these values are on opposite ends of the spectrum, where 1 in theory leads to overfitting and 21 to smoothing out details. Yet they provide equal and the best accuracy out of the tested values. 

Even though the assignment did not ask for it, I decided to test out normalizing the data first to see how it changes the prediction accuracy. I applied StandardScaler from sklearn and used the same k values as before. Surprisingly, I was able to achieve a much better accuracy:

- k = 1 --> Accuracy (scaled): 0.9444
- k = 5 --> Accuracy (scaled): 0.9444
- k = 11 --> Accuracy (scaled): 0.9444
- k = 15 --> Accuracy (scaled): 0.9722
- k = 21 --> Accuracy (scaled): 0.9444

However, this time, I achieved the best accuracy with k=15.

I then proceeded with the RNN algorithm with radius values as assigned in the assignment. However, I felt like the values were too high and quickly led to no changes to accuracy (most likely due to including all data values due to the radius being too large). I therefore decided to include a few smaller radius values.

- Radius = 50 --> Error: No neighbors found for test samples array([20]), you can try using a larger radius, giving a label for outliers, or considering removing them from your dataset.
- Radius = 100 --> Error: No neighbors found for test samples array([20]), you can try using a larger radius, giving a label for outliers, or considering removing them from your dataset.
- Radius = 150 --> Accuracy (unscaled): 0.8056
- Radius = 200 --> Accuracy (unscaled): 0.7222
- Radius = 250 --> Accuracy (unscaled): 0.7778
- Radius = 300 --> Accuracy (unscaled): 0.8056
- Radius = 350 --> Accuracy (unscaled): 0.7500
- Radius = 400 --> Accuracy (unscaled): 0.7222
- Radius = 450 --> Accuracy (unscaled): 0.7222
- Radius = 500 --> Accuracy (unscaled): 0.7222
- Radius = 550 --> Accuracy (unscaled): 0.7222
- Radius = 600 --> Accuracy (unscaled): 0.7222

Here we can see that the best accuracy was achieved with radius values of 150 and 300. 

Once again, I wanted to see what would happen with normalized data values. Once again, I used the StandardScaler and got the following values:

- Radius = 2.0 --> Error: No neighbors found for test samples array([ 1,  6, 13, 21, 22, 23, 27]), you can try using a larger radius, giving a label for outliers, or considering removing them from your dataset.
- Radius = 3.0 --> Accuracy (scaled): 0.9444
- Radius = 4.0 --> Accuracy (scaled): 1.0000
- Radius = 5.0 --> Accuracy (scaled): 0.9167
- Radius = 6.0 --> Accuracy (scaled): 0.5833
- Radius = 7.0 --> Accuracy (scaled): 0.4167
- Radius = 8.0 --> Accuracy (scaled): 0.3889
- Radius = 9.0 --> Accuracy (scaled): 0.3889
- Radius = 10.0 --> Accuracy (scaled): 0.3889

**This was probably the most interesting discovery. After normalizing the values, I got a perfect accuracy with Radius value 4, but then the accuracy started sharply declining after value 5. This unexpected behavior probably occurred because the different classes might be well separated from each other, and higher radius values might include data from other classes in the prediction. .**

## Visualizations

Here is the output from the model:


![KNN Plot](Screenshots/KNN%20Snapshot.png)



![RNN Plot](Screenshots/Unscaled%20RNN.png)



![RNN Scaled Plot](Screenshots/Scaled%20RNN.png)


