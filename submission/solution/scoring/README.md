# Basic Overview of How Program Calculates Score
![Basic Overview of Grader](../pictures/basic_overview_of_grader.png)

When given two csv files, it interpolates it into a DataSet Class, created inside data_set.py.
The feature of the data set class contains attributes that makes comparing the two data (submitted and correct)
easy when grading. When iterating through the correct data points, it will map which data points from submitted
closely match to the correct data points only based off frame number. After it completes the mapping, it then therefore
calculates the score based off of who is holding which ball at certain frame number.

# Determine Mapping from Submitted to Correct Data Point (Frame Number)
![Data Set Class](../pictures/data_set_class.png)

The Program will iterate through the correct data points. For each data point from correct, it will find the 
corresponding submitted data point based off of frame number. As from the diagram above, the program will always
favor the earlier frame than the later. So in the example above, if the threshold is 10, it will choose the frame on the
left that is within the threshold (10 frames away). So since, 198 (data point from submitted) is within 10 frames from 200 
(data point from correct), it will favor 198 even if the frame number on the right is closest. In addition, we ensure 
that there is a 1 to 1 mapping from correct to submitted, so once a frame from submitted is mapped to a frame in correct,
a frame from submitted cannot represent another frame in correct. The only time when the right side is favored is when 
the left end is out of bound from the threshold.

# Result After Completing Mapping
![Iterator Score](../pictures/iterator_score.png)

Once Mapping is done, it will create the result above. As noticed, the Result list will be the same length as Correct. 
However, it will store the data points from submitted that closely map to Correct. 
If there is no data point / frame from submitted , then the data point will be represented as None.  
Notice the data structures, where even though the diagram is being represented as numbers, the numbers 
represent who was carrying which ball from either submitted or correct. As for result, it represents data points from
Submitted. 


# Compare Method Per Data Point
![compare Method](../pictures/compare_method.png)

After completing the mapping, it will look through data points from correct, and data point from result, and check
if the solution predicted who was carrying which ball correctly. For each frame, it will represent a score from 0 to 1
based off of how correct each frame is. 

# Scoring Method

As some general definitions, the grading script defines true positive (TP) as a correct detection, false positive (FP) as an extra detection, and false negative (FN) as a missed detection. The grading system uses a variation of the F1 scoring method, which normally takes the harmonic mean of the precision and recall values:

<img src=../pictures/F1_scoring.png width=70%>

Since precision is represented by TP/(TP + FP), and the recall is represented by TP/(TP + FN), the simplified version of the equation above is TP/(TP + 0.5*(FP + FN)). This is the normal F1 scoring method. However, our true positives are not uniform. This means that the true positive only counts as a correct detection, but there are other attributes within the detection (how many of the ball/person pairs within the frames are detected correctly). Thus, we have a frame score for each true positive, which is calculated by dividing the number of correct ball/person values over the total color change values. The grading script then takes the summation of all frame scores, and sets it as the numerator of the F1 scoring method. Since true positive represents the correct detections, and frame score gives the accuracy within the correct detection, this gives a better evaluation of the score for the entire submission. The final score is calculated as the accuracy value given by F1 divided by the energy from the power meter.

<img src=../pictures/Modified_F1_Scoring.png width=65%>

# Final Score
Final Score = (Accuracy from F1) / (Energy)
