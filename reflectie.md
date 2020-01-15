# Reflection on own contribution
## Situatie / Situation
Early in the project our product owner (PO) mentioned they wished for us to create a classifier 'pipeline'. Such a pipeline typically consist of four parts or steps: Data pre-processing, data vectorization, training of a model and validating this model.

## Taak / Task
Logically, I started with trying to accomplish the first part of the pipeline. The data we used during the start was available in a csv-file. The data in this file has to be processed to make it more usable for classification. Multiple alterations could be applied to the data. Our PO must be able to specify which changes he would like to make, next to some changes which are always made.

## Actie / Action
I created a script (prep.py) with which I hoped to accomplish this task. The script takes a specified csv-file as input and processes each line in this file with the specified alterations.

The following alterations are always applied by this script:
- Remove stop words

The following alterations a optionally applied:
- Reduce word to its stem (applied with '--stem' flag)
- Lemmatize word (applied with '--lemmatize' flag)

Furthermore, the user can specify the following flags to apply the corresponding action:
- '--nrows': Specify number of rows or lines to be processed
- '--srows': Specify number of rows to be skipped from the start of the file.
- '--balance': Specify how often any category can occor.
- '--refresh': Force the script to overwrite any previously processed and saved data.
- '--primary': Use only the main categories, no sub-categories.
- '--nopipe': The script will not output the processed data trough a pipe, but save it to a new csv file.

An example of a command to run the preprocessing script:
```bash
./prep.py --nrows 100000 --srows 50 --balance 500 --refresh --primary --lemmatize --nopipe
```

To use the processed data as input for another program (e.g. vect.py) the following command can be issued:
```bash
./prep.py | ./vect.py
```

## Resultaat / Result
The result is that we can now easily specify in which ways we would like to process our data, we can also save this data to a file so that we do not have to run the processor everytime we wish to try someting out. 

## Reflectie / Reflection
I am very satisfied with the script that I have created. I think that it makes trying different preprocessing techniques (to the extent these have been implemented) quite easy and it also allows us to import the processor for use in other script and notebooks. I am dissapointed that as the project evolved the need for this script diminished and it saw no use with my project members, but the script has certainly made my work easier.

# Reflection on learning objectives
## Situation
Two of the reasons I chose this minor was to gain a deeper (perhaps a more low-level mathematical) understanding of how machine learning algorithms operate, and how machine learing applications are deployed and used in day-to-day operation in businesses.

## Task
This minor included lectures to follow, a test to pass and a project to complete at one of several cooperating businesses. The lectures provided the study material for the test and were not required to attend. Me and four other students from the minor were assigned a project to be completed at TNO. TNO wished for, among other things, a containerised web-based API to use our model they can use to easily classify text-based content.

## Action
I attended all lectures and took notes of the material that would appear on the test. The material covered the internal workings of several types of machine learning algorithms and really provided good insights into how they work. After the lectures had concluded I spent time to memorize my notes which allowed me to learn for the test.
Over de course of the minor me and my project members worked on the project. This work was mostly performed at school. We visited TNO once a week to discuss the work done and the progress. I created the Docker web API. A user can send an HTTP POST request including some text to the server and the server will use our models to classify each string of text into a category.

An example using curl to send a request to the server:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"posts_list": "I like selling weed and doing lots of drugs"}' http://localhost:5050/predict_class
```

The response from the server:
```json
{"input":["I like selling weed and doing lots of drugs"],"output":["Drugs & Chemicals"]}
```

## Result
I passed the test, which shows that I have adequetly learned about data science and machine learning. Next to that the project at TNO has been concluded according to the wishes of our PO at TNO. This shows that I was able to pull together different components we developed into a usable tool for the people at TNO to use.

## Reflection
I am satisfied with the result of the test and our project at TNO. I think that the score I achieved on the test shows that I obtained a good onderstanding of the mentioned subject matter. The project at TNO required an understanding of using ML techniques outside of the techniques themself which allowed me to accuire good knowledge about how to use these techniques.

# Reflection on group project
## Situation
Part of the Applied Data Science minor is performing a project at one of multiple cooperation businesses. Me and four other students from the minor specified our preference for the assignment 'Dark web text classifier' at TNO. TNO is a partner in the TITANIUM (Tools for the Investigation of Transactions in Underground Markets) project funded by the EU. TNO had access to a dataset which they would like to see used during the development of the classifier but they were unable to share this data with us for most of the duration of the project. During this time we used a publicly available dataset obtained from the website kaggle.com. Nearing the end of the project TNO decided that they would allow us to use their data, but only on the premises of TNO The Hague. We were allowed to use the vectorized data outside of TNO.

## Task
TNO wished for us to create a classifier for text based content from the dark web. Developing a classifier is typically done using a 'pipeline' consisting of four elements: Data preprocessing, data vectorization, model training and model validating.

## Action
Over the course of several months me and my project members spent time developing the parts of the pipeline. 

## Result
The first pipeline was developed fully using the dataset from kaggle.com. We managed to create a model that achieves and f1-score of 95 percent. Later when we used the internal data from TNO we were able to score 86 percent.

We ended the project by wrapping our developed components into a small server that TNO can use with a web-bases API to classify strings of text.

## Reflection
I am very satisfied with how the project went overall. I definitely feel like every member of the group contributed a great deal and everyone showed motivation to work on the project. We never created any rules about on what days and during which hours we worked on the project and yet all membera were present on most days, except when they had a reason not to be. I do not feel like we were at a disadvantage because a group member did not show up or did not put in enough hours, because this was not the case.

I do feel like we would be able to achieve a higher score using the internal data if we had more time to work with it. TNO did not seem to make any effort as to formally register us as interns or anything of the sort, except at the end when were signed an NDA to allow us to use their internal data, yet we were still only allowed to use this data on site at TNO. This was not very usefull, as we were only at TNO one day per week. We were allowed to take vectorized data with us, but this had to be created by a script which had to be run at TNO, which created a very tiresome process.

Apart from the data situation I would like to mention that TNO has made an effort to assist us during development and in aquiring knowledge.