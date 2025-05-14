In order to execute the app, one can either open it online with the following link:

https://evbuddy.streamlit.app


Alternatively, they can also execute the code locally if they have all the libraries listed in requirements.txt by executing the command streamlit run demo_app.py.

____________________________________________________________________________________________

This is the link to the GitHub repository: https://github.com/Gonzalo05/streamlit

____________________________________________________________________________________________


Here we will outline the project structure:


demo_app.py: this is the main app and where all the functionality takes place. The app was built in a single file, so there is more fluidity in the UX by dynamically loading different windows and functions.

requirements.txt: here we list the libraries we used to develop the app

styles.css: this is our custom CSS styling sheet. We insert this file into the app in demo_app to overwrite Streamlit's default styling.

contributionMatrix.pdf: this is the project's contribution matrix

.gitignore: Here we list patterns for files and directories that Git should ignore.

my_map_component: this folder is where we store the Streamlit component we build with React to make up for Streamlit's lack of capabilities when it comes to map interactions.

my_map_component/frontend/src/MyMap Component: this is the React file where we build the my_map_component functionality. 

ML_models: this folder stores the two ML models we built: ev_duration_prediction.joblib and ev_energy_prediction.joblib.

ML_Development_Files: here we store MlTrainingModel.ipynb, where we develop the ML models, and northcarolina.csv, which is a CSV containing the EV charging session data used in the ML. These files are just for documentation purposes and have no real impact on the app.

EV_data: here we store the data related to the EV car models and their specs. ev-data.json is from where we get all of the car specs and last_selection. JSON is where we save the car selection of the users.

.streamlit/config.toml: here we customise our Streamlit app settings. We used it to change the styling and theme of the app.