from fastapi import FastAPI
import uvicorn
from main import PredictRequest , predict_symptoms

app = FastAPI()

@app.get('/health')
def healthcheck():
  return {
      'status': 'ok'
  }

@app.post("/predict")
def predict(request: PredictRequest):
  try:
    result = predict_symptoms(request.gender, request.age, request.symptoms)
    return {
        "success":True,
        "input":{
            "gender":request.gender,
            "age":request.age,
            "symptoms":request.symptoms
        },
        "similiar_symptoms":result
    }
  except Exception as e:
    return {
        "success":False,
        "error":str(e)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)