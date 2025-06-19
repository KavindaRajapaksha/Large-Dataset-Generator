import os
import json
import random
import requests
from tqdm import tqdm
import time

# Set your GROQ API key
GROQ_API_KEY = "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"  # Replace with your actual API key

# Define topics and aspects for the agriculture Q&A
crops = ["rice", "tea", "coconut", "rubber", "vegetables", "fruits", "maize", "chili", "onion", 
         "potato", "cinnamon", "cardamom", "pepper", "cloves", "nutmeg", "sugarcane", "banana",
         "mango", "papaya", "jackfruit", "cassava", "sweet potato", "green gram", "black gram"]

animals = ["cows", "goats", "chickens", "ducks", "pigs", "buffalo", "bees", "fish", "shrimp",
           "sheep", "rabbits", "quail", "turkey", "cattle", "dairy cows", "beef cattle", 
           "freshwater fish", "carp", "tilapia", "catfish", "silkworms"]

aspects = ["farming", "diseases", "harvesting", "fertilizer", "irrigation", "climate adaptation", 
           "marketing", "best practices", "pest control", "sustainable methods", "organic farming",
           "processing", "storage", "yield improvement", "breeding", "feed management",
           "vaccination", "market prices", "export opportunities", "value addition",
           "traditional techniques", "modern technology", "crop rotation", "intercropping",
           "seasonal planning", "government subsidies", "loan schemes", "equipment", "water management"]

# Prompt templates
prompt_templates = [
    "Write a question and detailed answer about {crop} {aspect} in Sri Lanka.",
    "Generate a Q&A about {animal} {aspect} in Sri Lanka.",
    "What should a Sri Lankan farmer know about {aspect} in {crop} cultivation?",
    "How can {animal} farmers in Sri Lanka improve {aspect}?",
    "What are common {aspect} challenges for {crop} in Sri Lanka and how to address them?",
    "Provide a frequently asked question and answer about {animal} {aspect} in Sri Lanka.",
    "Generate a question and answer explaining {crop} {aspect} practices unique to Sri Lanka.",
    "Create a Q&A explaining how {aspect} affects {animal} farming in Sri Lanka's climate.",
    "What traditional knowledge about {crop} {aspect} exists in Sri Lankan farming communities?",
    "How does Sri Lanka's climate influence {aspect} practices for {animal} farming?",
    "Generate a detailed Q&A about economic aspects of {crop} {aspect} in Sri Lanka.",
    "Create a Q&A about how small-scale farmers handle {animal} {aspect} in Sri Lanka."
]

def generate_prompts(n):
    """Generate unique prompts by combining templates with topics"""
    prompts = set()
    while len(prompts) < n:
        template = random.choice(prompt_templates)
        if "{crop}" in template and "{aspect}" in template:
            crop = random.choice(crops)
            aspect = random.choice(aspects)
            prompts.add(template.format(crop=crop, aspect=aspect))
        elif "{animal}" in template and "{aspect}" in template:
            animal = random.choice(animals)
            aspect = random.choice(aspects)
            prompts.add(template.format(animal=animal, aspect=aspect))
        elif "{aspect}" in template and "{crop}" in template:
            crop = random.choice(crops)
            aspect = random.choice(aspects)
            prompts.add(template.format(crop=crop, aspect=aspect))
    return list(prompts)

def get_qa_from_llm(prompt):
    """Query GROQ API with Llama 3 70B model to generate Q&A"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert Sri Lankan agriculture assistant. For each prompt, generate a clear question and a detailed, helpful answer about agriculture or animal farming in Sri Lanka. Format your response as a JSON object with 'input' (question) and 'output' (answer) keys."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"]
            
            # Try to extract JSON directly
            try:
                # Find JSON object in the response
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx]
                    qa = json.loads(json_str)
                    if "input" in qa and "output" in qa:
                        return qa
            except:
                pass
                
            # If direct JSON extraction fails, try parsing from text
            try:
                lines = text.strip().split("\n")
                input_line = None
                output_lines = []
                capture_output = False
                
                for line in lines:
                    line = line.strip()
                    if line.lower().startswith(("input:", "question:")):
                        input_line = line.split(":", 1)[1].strip()
                    elif line.lower().startswith(("output:", "answer:")):
                        capture_output = True
                        output_text = line.split(":", 1)[1].strip()
                        if output_text:
                            output_lines.append(output_text)
                    elif capture_output and line:
                        output_lines.append(line)
                
                if input_line and output_lines:
                    return {
                        "input": input_line,
                        "output": " ".join(output_lines)
                    }
            except:
                pass
        else:
            # Handle rate limiting
            if response.status_code == 429:
                time.sleep(5)  # Wait 5 seconds before retrying
                
        return None
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def main():
    # Generate about 2100 prompts to ensure we get at least 2000 unique Q&A pairs
    prompts = generate_prompts(2100)
    qa_pairs = []
    seen_inputs = set()
    
    print(f"Generating {len(prompts)} prompts to get at least 2000 unique Q&A pairs...")
    
    for prompt in tqdm(prompts, desc="Generating Q&A pairs"):
        qa = get_qa_from_llm(prompt)
        
        # Check if we got a valid response and it's not a duplicate
        if qa and "input" in qa and "output" in qa:
            # Clean the data
            qa["input"] = qa["input"].strip()
            qa["output"] = qa["output"].strip()
            
            if qa["input"] not in seen_inputs and qa["input"] and qa["output"]:
                qa_pairs.append(qa)
                seen_inputs.add(qa["input"])
                
                # Occasionally save progress
                if len(qa_pairs) % 100 == 0:
                    with open("sri_lanka_agriculture_qa_temp.json", "w", encoding="utf-8") as f:
                        json.dump(qa_pairs, f, ensure_ascii=False)
                    print(f"Progress: {len(qa_pairs)} unique Q&A pairs generated")
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(1)
        
        # Check if we've reached our target
        if len(qa_pairs) >= 2000:
            break
    
    print(f"Generated {len(qa_pairs)} unique Q&A pairs")
    
    # Save the final dataset
    with open("sri_lanka_agriculture_qa.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False)
    
    print("Dataset saved to sri_lanka_agriculture_qa.json")

if __name__ == "__main__":
    main()
