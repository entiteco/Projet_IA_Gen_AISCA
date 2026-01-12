import pandas as pd
import os

def jobs_filter():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        jobs_path = os.path.join(base_dir, "raw_IT_jobs", "JobsDatasetProcessed.csv")
        output_path_json = os.path.join(base_dir, "filtered_it_jobs.json")

        jobs = pd.read_csv(
            jobs_path,
            sep=',',
            encoding='utf-8',
            on_bad_lines='skip'
        )

        columns_map = {
            "ID": "id", 
            "Job Title": "job_title", 
            "Description": "description", 
            "IT Skills": "it_skills", 
            "Soft Skills": "soft_skills",
            "Education": "education"
        }

        filtered_jobs = jobs[list(columns_map.keys())].rename(columns=columns_map)

        for col in ['it_skills', 'soft_skills']:
            filtered_jobs[col] = filtered_jobs[col].fillna("").str.split(',').apply(lambda x: [i.strip() for i in x if i.strip()])

        filtered_jobs.to_json(
            output_path_json, 
            orient='records', 
            indent=4, 
            force_ascii=False
        )
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    jobs_filter()