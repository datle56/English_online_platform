# from fastapi import FastAPI
# from pydantic import BaseModel
# from gector import predict, load_verb_dict
# from transformers import AutoTokenizer
# import torch
# from grammar_check import llm_feedback
# # Load resources and model at startup
# encode, decode = load_verb_dict('data/verb-form-vocab.txt')
# model = torch.load('gector-deberta-v3.pth')
# tokenizer = AutoTokenizer.from_pretrained('token')

# if torch.cuda.is_available():
#     model.cuda()

# app = FastAPI()

# # Define input and output models
# class CorrectionRequest(BaseModel):
#     text: str

# class CorrectionResponse(BaseModel):
#     explanation : str
#     corrected_sentence: str

# @app.post("/correct", response_model=CorrectionResponse)
# async def correct_text(request: CorrectionRequest):
#     # Convert the input text (str) into a list with one sentence
#     srcs = [request.text]

#     predict_args = {
#         'model': model,
#         'tokenizer': tokenizer,
#         'srcs': srcs,
#         'encode': encode,
#         'decode': decode,
#         'keep_confidence': 0,
#         'min_error_prob': 0,
#         'batch_size': 1,
#         'n_iteration': 5
#     }

#     # Generate corrected sentences
#     final_corrected_sents = predict(**predict_args)
    
#     # Join the result back into a single string
#     corrected_text = final_corrected_sents[0] if final_corrected_sents else ""
#     print(corrected_text)
#     explanation, corrected_sentence = llm_feedback(input_sentence= request.text, output_sentence= corrected_text)
#     return CorrectionResponse(explanation=explanation, corrected_sentence=corrected_sentence)

# # Run with: uvicorn filename:app --reload
