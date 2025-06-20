import os
import json
import random
import requests
from tqdm import tqdm
import time

# Set your GROQ API key
GROQ_API_KEY = "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"

# Define Sri Lankan tourism topics and aspects
places = [
    "Sigiriya", "Galle Fort", "Ella", "Nuwara Eliya", "Kandy", "Yala National Park",
    "Anuradhapura", "Polonnaruwa", "Jaffna", "Trincomalee", "Bentota", "Mirissa",
    "Arugam Bay", "Hikkaduwa", "Colombo", "Dambulla", "Pinnawala", "Udawalawe",
    "Knuckles Range", "Horton Plains", "Adam's Peak", "Negombo", "Kalpitiya",
    "Batticaloa", "Mannar", "Sinharaja Forest", "Wilpattu", "Matara", "Hambantota"
]

aspects = [
    "travel tips", "best time to visit", "local food", "cultural significance",
    "things to do", "adventure activities", "heritage", "religious importance",
    "transportation", "historical facts", "eco-tourism", "wildlife",
    "accommodation", "festivals", "beaches", "UNESCO sites", "photography spots",
    "cost of travel", "climate", "local customs", "shopping", "surfing",
    "safety tips", "must-see attractions", "family-friendly spots",
    "budget travel", "luxury travel", "sustainable tourism"
]

prompt_templates = [
    "Generate a question and detailed answer about {place} {aspect} in Sri Lanka.",
    "What should a tourist know about {aspect} when visiting {place} in Sri Lanka?",
    "Write a travel Q&A on {place} and its {aspect}.",
    "Explain {aspect} for tourists planning to visit {place}, Sri Lanka.",
    "Create a Q&A about how to experience {place}'s {aspect} in an authentic way.",
    "What makes {place} in Sri Lanka special in terms of {aspect}?",
    "Describe a common tourist question and answer about {place} related to {aspect}.",
    "How does Sri Lanka’s culture influence the experience of {aspect} in {place}?",
    "Give a tourist guide Q&A focusing on {place} and its {aspect}.",
    "Provide a Q&A about unique {aspect} found in {place}, Sri Lanka."
]

def generate_prompts(n):
    prompts = set()
    while len(prompts) < n:
        template = random.choice(prompt_templates)
        place = random.choice(places)
        aspect = random.choice(aspects)
        prompts.add(template.format(place=place, aspect=aspect))
    return list(prompts)

def get_qa_from_llm(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a tourism expert on Sri Lanka. For each prompt, generate a clear question and detailed, informative answer about tourism in Sri Lanka. Format the response as a JSON with 'input' and 'output' keys."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"]
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    qa_json = text[start:end]
                    qa = json.loads(qa_json)
                    if "input" in qa and "output" in qa:
                        return qa
            except:
                pass

            # Fallback: extract manually
            try:
                lines = text.strip().split("\n")
                input_line = ""
                output_lines = []
                capture = False

                for line in lines:
                    if line.lower().startswith(("input:", "question:")):
                        input_line = line.split(":", 1)[1].strip()
                    elif line.lower().startswith(("output:", "answer:")):
                        capture = True
                        output_text = line.split(":", 1)[1].strip()
                        if output_text:
                            output_lines.append(output_text)
                    elif capture:
                        output_lines.append(line.strip())

                if input_line and output_lines:
                    return {"input": input_line, "output": " ".join(output_lines)}
            except:
                pass

        elif response.status_code == 429:
            time.sleep(5)
        return None
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def clean_qa(qa):
    """Ensure input and output are strings and stripped"""
    if "input" in qa:
        qa["input"] = str(qa["input"]).strip()

    if "output" in qa:
        if isinstance(qa["output"], str):
            qa["output"] = qa["output"].strip()
        elif isinstance(qa["output"], dict):
            qa["output"] = " ".join(str(v).strip() for v in qa["output"].values())
        else:
            qa["output"] = str(qa["output"]).strip()

    return qa

def main():
    prompts = generate_prompts(2100)
    qa_pairs = []
    seen_inputs = set()

    print(f"Generating {len(prompts)} prompts to get at least 2000 unique tourism Q&A pairs...")

    for prompt in tqdm(prompts, desc="Generating Q&A pairs"):
        qa = get_qa_from_llm(prompt)
        if qa and "input" in qa and "output" in qa:
            qa = clean_qa(qa)
            if qa["input"] not in seen_inputs and qa["input"] and qa["output"]:
                qa_pairs.append(qa)
                seen_inputs.add(qa["input"])

                if len(qa_pairs) % 100 == 0:
                    with open("sri_lanka_tourism_qa_temp.json", "w", encoding="utf-8") as f:
                        json.dump(qa_pairs, f, ensure_ascii=False)
                    print(f"Saved {len(qa_pairs)} Q&A so far...")

        time.sleep(1)
        if len(qa_pairs) >= 2000:
            break

    print(f"\n✅ Completed: {len(qa_pairs)} Q&A pairs generated.")
    with open("sri_lanka_tourism_qa.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False)
    print("📁 Dataset saved to sri_lanka_tourism_qa.json")

if __name__ == "__main__":
    main()
