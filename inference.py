from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration 

# Load the three open-source LLM models with small parameter size
model_1_name = "google/codegemma-2b"
model_2_name = "google/flan-t5-small"
model_3_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load the fourth model, which will summarize the answers
summary_model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Summarizer model (up-to-date)

# Load models and their tokenizers
tokenizer_1 = AutoTokenizer.from_pretrained(model_1_name)
model_1 = AutoModelForCausalLM.from_pretrained(model_1_name)

tokenizer_2 = AutoTokenizer.from_pretrained(model_2_name)
model_2 = T5ForConditionalGeneration.from_pretrained(model_2_name)

tokenizer_3 = AutoTokenizer.from_pretrained(model_3_name)
model_3 = AutoModelForCausalLM.from_pretrained(model_3_name)

summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
summary_model = AutoModelForCausalLM.from_pretrained(summary_model_name)

# Helper function to generate answer from each model
def generate_answer(model, tokenizer, question, max_length=1000):
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 1. Receive an input question from the user
# "If it takes 3 towels 3 hours to dry in the sun, how long will it take 5 towels to dry?"
user_question = input("Enter your question: ")

# 2. Collect output from the three open-source models
answer_1 = generate_answer(model_1, tokenizer_1, user_question)
answer_2 = generate_answer(model_2, tokenizer_2, user_question)
answer_3 = generate_answer(model_3, tokenizer_3, user_question)

# 3. Fuse the three answers in a single answer with the specified format
fused_answer = f"""*** Answer 1:
{answer_1}
*** Answer 2:
{answer_2}
*** Answer 3:
{answer_3}
"""

# Print the fused answer
print("\nFused Answer from the three models:\n")
print(fused_answer)

# 4. Ask the fourth model (summarizer) to summarize the fused answer
# In this case, we'll use LLaMA 3.2 for summarization.
# We create a system prompt instructing the model to summarize the inputs and remove errors.

system_prompt = "You have been provided with a set of responses from various large language models to a user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Do not add any extra comments about how you created the response, just synthesize these responses as instructed."

# Combine the system prompt with the fused answer
summary_input = f"{system_prompt}\n\n{fused_answer}"

# Using the summarization model to generate a final output
summary_inputs = summary_tokenizer.encode(summary_input, return_tensors="pt")
summary_outputs = summary_model.generate(summary_inputs, max_length=3000, num_return_sequences=1, pad_token_id=summary_tokenizer.eos_token_id)
summary = summary_tokenizer.decode(summary_outputs[0], skip_special_tokens=True)

# 5. Print the final summarized response from the fourth model
print("\nFinal summarized answer from the fourth model:\n")
print(summary)