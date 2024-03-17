import pandas as pd
import time

def predict_batch(model, data, prompt, batch_size=30):
    ratings = []
    for i in range(0, data.shape[0] - batch_size + 1, batch_size):
        try:
            review_id = data.iloc[i:i+batch_size]['review_id'].tolist()
            review_text = data.iloc[i:i+batch_size]['review_text'].tolist()
            user_input = "\n".join([f"{_id}|{txt}" for _id, txt in zip(review_id, review_text)])
            prediction = model.invoke(prompt.format(user_input))
            ratings.append(prediction.content)
            time.sleep(1)
        except Exception as e:
            print(f"Error while predicting batch: {e}")
    return ratings

def parse_model_predicted_ratings(raw_predictions):
    predicted_ratings = []
    for r in raw_predictions:
        individual_ratings = r.split("\n")
        for i in individual_ratings:
            try:
                predicted_ratings.append(
                    {
                        'review_id': int(i.split("|")[0]),
                        'rating': int(i.split("|")[1])
                    }
                )
            except ValueError as e:
                continue
    return pd.DataFrame(predicted_ratings)