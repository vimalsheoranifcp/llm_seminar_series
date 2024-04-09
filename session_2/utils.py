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
                review_id = i.split("|")[0]
                rating = i.split("|")[1]
                if not (0 < rating < 10):
                    raise ValueError(f"Invalid rating: {rating} for rating string '{i}' for raw prediction {r}")
                predicted_ratings.append({'review_id': review_id, 'rating': rating})
            except ValueError as e:
                print(f"Error while parsing predicted rating: {e}")
                continue
    return pd.DataFrame(predicted_ratings)

def parse_cot_output(output, start_tag="<rating>", end_tag="</rating>"):
    try:
        start = output.find(start_tag)
        if start == -1:
            raise ValueError("Couldn't find start tag.")
        start_tag_length = len(start_tag)
        end = output.find(end_tag, start + start_tag_length)
        if end == -1:
            raise ValueError("Couldn't find end tag.")
        return int(output[start+start_tag_length:end])
    except Exception as e:
        print(f"Error while parsing output: {e}\nOutput:{output}")
        return -1

def cot_parsing_chain(review, prompt, llm, output_parser):
    try:
        output = llm.invoke(prompt.format(review))
        final_rating = output_parser(output.content)
        return final_rating
    except Exception as e:
        print(f"Error while generating prediction: {e}")
        return -1
    
def get_combined_rating(ratings):
    non_erroneous_ratings = [rating for rating in ratings if rating != -1]
    if len(non_erroneous_ratings) == 0:
        raise ValueError("All the ratings are erroneous")
    return round(sum(non_erroneous_ratings) / len(non_erroneous_ratings))

async def sc_parsing_chain(review, prompt, llm, output_parser, samples=5):
    try:
        outputs = await llm.abatch([prompt.format(review) for _ in range(samples)])
        parsed_outputs = [output_parser(output.content) for output in outputs]
        return get_combined_rating(parsed_outputs), parsed_outputs
    except Exception as e:
        print(f"Error while generating prediction: {e}")
        return -1, []