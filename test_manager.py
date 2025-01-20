import os
import time
import json
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Task 1: Dynamic Data Scraping
def scrape_jobs():
    # Setup WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    driver_path = "desktop/ksz/chromedriver"  # Update with your path
    driver = webdriver.Chrome(service=Service(driver_path), options=chrome_options)

    url = "https://www.linkedin.com/jobs/search/"
    driver.get(url)


    jobs = []

    try:
        for page in range(1, 6):  

            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "jobs-search-results__list-item"))
            )

          
            job_cards = driver.find_elements(By.CLASS_NAME, "jobs-search-results__list-item")

            for card in job_cards:
                job_title = card.find_element(By.CLASS_NAME, "job-card-list__title").text
                company_name = card.find_element(By.CLASS_NAME, "job-card-container__company-name").text
                location = card.find_element(By.CLASS_NAME, "job-card-container__metadata-item").text
                date_posted = card.find_element(By.CLASS_NAME, "job-card-container__metadata-item--date").text

         
                try:
                    card.click()
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "jobs-description__content"))
                    )
                    job_description = driver.find_element(By.CLASS_NAME, "jobs-description__content").text
                except Exception:
                    job_description = ""

                jobs.append({
                    "Job Title": job_title,
                    "Company Name": company_name,
                    "Location": location,
                    "Date Posted": date_posted,
                    "Job Description": job_description,
                })

            try:
                next_button = driver.find_element(By.CLASS_NAME, "artdeco-pagination__button--next")
                next_button.click()
                time.sleep(2)  
            except Exception:
                break

    finally:
        driver.quit()

   
    with open("jobs_data.json", "w") as file:
        json.dump(jobs, file, indent=4)

    print("Data scraping completed and saved to jobs_data.json.")


def clean_data():

    with open("jobs_data.json", "r") as file:
        raw_data = json.load(file)

    df = pd.DataFrame(raw_data)


    df.drop_duplicates(inplace=True)

    df["Job Description"] = df["Job Description"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)


    keywords = ["python", "ai", "data science"]
    for keyword in keywords:
        df[keyword] = df["Job Description"].str.contains(keyword, na=False).astype(int)

    df["Job Posting Age"] = df["Date Posted"].apply(lambda x: calculate_age(x))
    df["Job Type"] = df["Job Description"].apply(lambda x: "Remote" if "remote" in x else "In-Office")


    df.to_csv("cleaned_jobs_data.csv", index=False)
    print("Data cleaning completed and saved to cleaned_jobs_data.csv.")

    

# Helper function to calculate job posting age
# def calculate_age(date_posted):
#     # Assuming date_posted is in format "x days ago"
#     try:
#         days = int(date_posted.split(" ")[0])
#         return days
#     except ValueError:
#         return None

# Task 3: Data Analysis
def analyze_data():
    df = pd.read_csv("cleaned_jobs_data.csv")

  
    top_cities = df["Location"].value_counts().head(5)
    print("Top 5 Cities:\n", top_cities)

    skill_counts = df[["python", "ai", "data science"]].sum()
    print("Skill Trends:\n", skill_counts)

    top_companies = df["Company Name"].value_counts().head(5)
    print("Top Companies:\n", top_companies)

    #df["Job Posting Age"].plot(kind="hist", bins=10, title="Job Age Distribution")
    #plt.show()


def advanced_insights():
    df = pd.read_csv("cleaned_jobs_data.csv")

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    df["Sentiment"] = df["Job Description"].apply(lambda x: sia.polarity_scores(x)["compound"])

    # Predict Job Popularity
    features = df[["python", "ai", "data science"]]
    target = df["Job Posting Age"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Model trained. Predictions:\n", predictions)


def visualize_data():
    df = pd.read_csv("cleaned_jobs_data.csv")

    df["Location"].value_counts().head(5).plot(kind="bar", title="Top 5 Cities")
    plt.show()

    text = " ".join(df["Job Description"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
 
 #pie chart
    df["Job Type"].value_counts().plot(kind="pie", autopct="%1.1f%%", title="Job Type Distribution")
    plt.show()

    # Scatter plot=====>>>>>>>>>>>>>>>: Job popularity vs. Skills Count
    df.plot.scatter(x="python", y="Job Posting Age", title="Popularity vs. Skills")
    plt.show()

# Execute all tasks
if __name__ == "__main__":
    scrape_jobs()
    clean_data()
    analyze_data()
    advanced_insights()
    visualize_data()
