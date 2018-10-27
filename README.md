# CS-433 Machine Learning - Project 1
Project in CS-433 - Machine Learning, EPFL (Fall 2018).

Group: Maiken Berthelsen, Ida Sandsbraaten and Sigrid Wanvik Haugen.




### Running instructions:
- Download the .zip-file containing the data set from https://www.kaggle.com/c/epfml18-higgs/data. Make sure that the folder containing these files are in the same folder as run.py

- The code can be run by calling "python main.py" from your terminal

- Her bør vi legge til noe som printes til terminal når det er ferdig

- The submission csv file can be found in the same folder with the name "Results.csv"




### Code architecture:
This code contains the following files:
* run.py 

	The whole code can be executed by running run.py. This file imports helpers.py and implementations.py

* helpers.py

	Contains the given functions for loading csv, predicting labels and create a csv submission.

* implementations.py

	Contains all of our implementations and is divided into three parts, each containing:

		- The six ML methods that are asked for in the project description.

		- The functions for data processing.

		- Contains the functions used in the ML methods.

