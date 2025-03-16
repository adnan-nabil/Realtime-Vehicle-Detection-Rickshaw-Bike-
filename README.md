# Rickshaw and Bike Detection

1. This is a PyQt project. So, here we can input a video, then model can detect rickshaw and bike realtime. 
2. Dataset: I collected images of dhaka city traffic from google search.
3. And then annotate with the bounding box using Labelme tool.
4. I did augmentation for increase the number of image data. For data augmentation I used albumentations.
5. For model, I have used here YOLO v8 model.


# Accuracy
Overall accuracy of the model is 85.6%

Precision: Bike = 80 %

           Rickshaw = 85.5 %
           
           Background = 50 %
           
Recall: Bike = 66.7 %

        Rickshaw = 89 %
        
        Background = 61.5 %


We can use this type of project for detecting wrong side traffic movement of rickshaw and bike in dhaka city.        
