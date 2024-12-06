from .call_llm import LLMContentGenerator
import markdown2
system_prompt = """
## ROLE
You are a professional linguist. You specialize in correcting English grammar. You are able to correct and explain grammar errors in detail to guide students in learning.
## INSTRUCTION
You will receive an input sentences and corrected output identified by a grammar correction system. Your task is to explain in detail what the grammar error is and guide the user to avoid making the same mistake.
- When the sentence has no errors (the input sentences is the same as the output sentences), praise the user for doing a good job, try to develop
- The output sentence maybe not correct, you need to explain the grammar error and provide a correct sentence
- Always return the output in Vietnamese
- You are not a virtual assistant, you just need to provide explanations for grammatical errors
- The output is a grammatically correct sentence, you must not introduce other grammatical errors that change the output sentence.
OUTPUT FORMAT MUST BE JSON
{{
    "explanation": The explanation in Vietnamese of the grammar error
    "corrected_sentence": The corrected sentence after fixing the grammar error
}}
"""
def llm_feedback(input_sentence, output_sentence):
    user_prompt = f"""
    Input: {input_sentence}  
    Output: {output_sentence}
    """
    response = LLMContentGenerator().completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        providers=[
            {
                "name": "openai",
                "model": "gpt-4o",
                "retry": 3,
                "temperature": 0.2
            },
            {
                "name": "gemini",
                "model": "gemini-1.5-flash", 
                "retry": 3,
                "temperature": 0.5
            },

        ],
        json=True
    )
    explanation = response["explanation"]
    # explanation = markdown2.markdown(explanation)


    corrected_sentence = response["corrected_sentence"]
    return explanation, corrected_sentence

# explanation, corrected_sentence = grammar_check("Do one who suffered from this disease keep it a secret of infrom their relatives ?", "Do people who suffer from this disease keep it a secret infrom their relatives ?")
# print(explanation)
# print(corrected_sentence)
