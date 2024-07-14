#to run the application use command flask --app appv6 run
#this is an updated version of the appv5 (updated after 23rd of April)
#some coding in this part has been inspired from the pymoo official website availble at: https://pymoo.org/index.html

import pymoo 
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


course_data = pd.read_csv("Coursera.csv") # loadig data from csv file. course_data is the dataframe
course_data = course_data[course_data['Course Rating']!="Not Calibrated"] # removing  not calibrated form the data set
course_data = course_data.head(500) # we can select fewer records rather than the whole dataset

course_data['Course Rating'] = course_data['Course Rating'].astype(float) # getting the float
data = course_data[['Course Name','Difficulty Level','Course Description','Skills']] # we have selected just (Course Name','Difficulty Level','Course Description','Skills') column

course_data.index = course_data['Course Name']# we are storing the indexing the courses
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Size' column
course_data['Difficulty'] = label_encoder.fit_transform(course_data['Difficulty Level'])
course_data.index = course_data['Course Name']
course_data=course_data.to_dict('records')# to convert out data into rocodes so that we can go thoug each record in dataframe and create didctionay
new_dict = {}
for i in (course_data):# looping throught the records and creating dictioanry 
    new_dict[i['Course Name'], i['Skills'], i['Difficulty Level'], i['Course Rating'], i['Course URL']] = i #comment and uncomment if you need to change the output display
    #new_dict[i['Course Name'], i['Course Description'], i['Difficulty Level'], i['Course Rating'], i['Course URL']] = i

courses_data = new_dict
#comment and uncomment where need to search from description or skills (preferably skills)

course_skills = [data["Skills"] for data in courses_data.values()]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(course_skills)

'''
course_descriptions = [data["Course Description"] for data in courses_data.values()]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(course_descriptions)

'''

# Define the course recommendation problem
def cosine_similarity_function(x, skills):
    keywords_vector = tfidf_vectorizer.transform([skills])
    return -sum(x*cosine_similarity(keywords_vector, tfidf_matrix).flatten())


class CourseRecommendationProblem(ElementwiseProblem):# ElementwiseProblem is defined in pymoo we are inheriting from ElementwiseProblem

    def __init__(self, skills, rating, difficulty):# __init__ is constructor in python
        self.skills = skills
        self.rating = rating
        self.difficulty = difficulty
        xl = np.full(len(courses_data), 0.0)
        xu = np.full(len(courses_data), 1.0)
        #using super we are calling ElementwiseProblem class constructor
        super().__init__(
            n_var=len(courses_data), n_obj=1, n_constr=0, xl=xl, xu =xu)
    #_evaluate is a overridden funciton
    def _evaluate(self, x, out, *args, **kwargs):
        #distances = np.array([[np.linalg.norm(np.array([x[i][0], x[i][1]]) - np.array([courses_data[course]["rating"], courses_data[course]["difficulty"]])) for i in range(len(x))] for course in courses_data])
        #out["F"] = np.column_stack([np.min(distances, axis=0), np.mean(distances, axis=0), np.max(distances, axis=0), cosine_similarity_function(x, keyword)])

        out["F"] = cosine_similarity_function(x, self.skills)# Define the course recommendation problem instance



def get_recommended_course(skills, rating, difficulty):
    algorithm = NSGA2(pop_size=100)
    problem = CourseRecommendationProblem(skills, rating, difficulty)# creating object of CourseRecommenderProblem
    res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=True,
               save_history=True,)
    subject_requirements = [skills, rating, difficulty]  # Example requirements (skills, rating, difficulty level)
    if res.X.ndim>1:
        a = res.X[0]
    else:
        a = res.X
    print(res.X)
    print("\n\n\n\n\n length of Res ", len(res.X),"\n\n\n\n\n\n") 
    ans = sorted(range(len(a)), key=lambda i: a[i],reverse=True)[:5]
    recommend_courses = []
    for i,x in enumerate(new_dict):
        if i in ans:
            recommend_courses.append(x)
    return recommend_courses

#Define the route for the landing page of flask app/ default rout
@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

# Define the Flask route for recommending courses based on keywords
@app.route('/recommend_courses/',  methods=['POST'])
def recommend_courses():
    skills = request.form['skills']
    rating = float(request.form['rating'])
    difficulty = float(request.form['difficulty'])     
    recommended_courses = get_recommended_course(skills, rating, difficulty)
    return render_template('index.html',taskslist=recommended_courses)

    #res = jsonify({'recommended_courses': recommended_courses})
    #return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)