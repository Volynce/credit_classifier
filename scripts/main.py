import subprocess

def run_data_preprocessing():
    print("Запуск data_preprocessing.py...")
    subprocess.run(["python", "data_preprocessing.py"])

def run_model_training():
    print("Запуск model_training.py...")
    subprocess.run(["python", "model_training.py"])

def run_evaluation():
    print("Запуск evaluation.py...")
    subprocess.run(["python", "evaluation.py"])

if __name__ == "__main__":
    run_data_preprocessing()
    run_model_training()
    run_evaluation()
