from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI()

interpreter = tf.lite.Interpreter(model_path="model/student_regression_dl.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load("model/scaler.save")


class StudentInput(BaseModel):
    hours_studied: int = Field(..., description="Jumlah jam belajar")
    previous_scores: int = Field(..., description="Nilai sebelumnya")
    extracurricular_activities: int = Field(..., description="1 jika ikut ekstrakurikuler, 0 jika tidak")
    sleep_hours: int = Field(..., description="Jam tidur per hari")
    sample_question_papers_practiced: int = Field(..., description="Jumlah soal latihan dikerjakan")


@app.post("/predict")
def predict_performance(data: StudentInput):
    raw_features = np.array([[
        data.hours_studied,
        data.previous_scores,
        data.extracurricular_activities,
        data.sleep_hours,
        data.sample_question_papers_practiced
    ]], dtype=np.float32)

    scaled_features = scaler.transform(raw_features).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], scaled_features)

    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    return {
        "predicted_performance_index": round(float(prediction), 2)
    }
