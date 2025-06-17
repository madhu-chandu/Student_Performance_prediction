# Student Performance Prediction 🎓

This project predicts students' math scores using machine learning. It uses Python, pandas, scikit-learn, and Flask to build a full data science pipeline from cleaning to web deployment.

## Project Structure

Student_Performance_Project/
├── data/
│   └── StudentsPerformance.csv
├── output/
│   ├── math_score_model.pkl
│   ├── le_gender.pkl
│   ├── le_race_ethnicity.pkl
│   ├── le_parental_level_of_education.pkl
│   ├── le_lunch.pkl
│   └── le_test_preparation_course.pkl
├── visuals/
│   └── math_score_dist.png
├── source/
│   ├── main.py
│   └── app.py
├── templates/
│   └── index.html
└── README.md

## Technologies Used

- Python 3.x
- pandas
- matplotlib, seaborn
- scikit-learn
- Flask
- joblib

## Dataset

The dataset includes features like gender, race/ethnicity, parental education, lunch type, test preparation, and exam scores.

Source: Kaggle - Student Performance in Exams

## Steps to Run

1. Clone the repository

git clone https://github.com/madhu-chandu/student-performance-predictor.git

2. Install the required packages

pip install pandas matplotlib seaborn scikit-learn flask joblib

3. Train the model

cd source
python main.py

4. Start the Flask app

python app.py

Then go to http://127.0.0.1:5000 in your browser.

## How It Works

- Cleans and encodes the dataset
- Trains a Linear Regression model to predict math scores
- Saves the model and encoders
- Provides a web UI to input student data and get math score predictions

## Output Example

R² Score: 0.85 (example)  
Mean Squared Error: 42.7 (example)

## License

MIT License
