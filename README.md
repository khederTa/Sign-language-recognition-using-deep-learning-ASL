# Sign-language-recognition-using-deep-learning-ASL
Work in this project was to build an application to help the deaf and mute to communicate with other people easily to meet their needs and help them integrate into society. Through this application, the signal that the person puts in front of the camera is predicted and the corresponding letter is displayed in order to formulate a sentence that expresses what this person is saying. 
Dataset link: "https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train"
In this project we use MobileNet, which is a type of transfer learning network, as it is a pre-trained network on a data set called ImageNet. It is a lightweight and faster network because it reduces the calculations in one layer by 9 times less and separates the filter application process in a layer and the merge process in a second layer to create New features.

**Fine-Tuning** means taking the weights of a trained neural network and using it as an initialization for a new model that is trained on data from the same domain.
It is used for:
1- Training Acceleration
2- Overcome small data set size

There are different strategies, such as training the fully configured network or "freezing" some pre-trained weights (usually whole layers).
