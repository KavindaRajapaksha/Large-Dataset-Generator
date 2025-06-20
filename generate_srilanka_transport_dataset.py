import os
import json
import random
import requests
from tqdm import tqdm
import time

GROQ_API_KEY = "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"

transport_types = [
    "public bus", "intercity bus", "private taxi", "tuk tuk", "train", "express train",
    "ride-sharing", "airport shuttle", "domestic flight", "international flight",
    "cargo ship", "passenger ferry", "harbor transport", "rental car", "motorbike rental",
    "bicycle rental", "three-wheeler", "long distance bus", "luxury coach",
    "tourist van", "airport transfer", "boat service", "fishing boat", "highway bus",
    "night train", "sleeper bus", "VIP taxi", "school bus", "minibus", "tourist bus"
]
transport_aspects = [
    "fares", "routes", "ticket booking", "comfort", "safety", "accessibility", "availability",
    "popular destinations", "travel time", "operating hours", "reliability", "air conditioning",
    "crowd levels", "frequency", "how to book", "online booking", "payment methods",
    "language barriers", "tips for tourists", "local experience", "group travel",
    "luggage policy", "pet policy", "child friendly", "elderly friendly", "eco-friendliness",
    "unique features", "best for families", "private hire", "shared rides", "connecting services"
]
air_transport_aspects = [
    "international routes", "airlines operating from Sri Lanka", "airport lounges", "baggage rules",
    "flight booking tips", "SriLankan Airlines", "visa requirements for outbound flights",
    "duty free shopping", "airport transfer options", "connecting flights", "seat selection",
    "onboard services", "direct flights", "stopovers", "flight delays", "airfare deals"
]
ship_transport_aspects = [
    "international routes", "major ports", "passenger ferries from Sri Lanka", "cruise lines",
    "cargo shipping", "customs procedures", "ferry schedules", "booking tickets", "onboard amenities",
    "sea travel safety", "visa requirements", "baggage policy", "eco-friendly sea travel",
    "port facilities", "connecting transport", "embarkation process", "ferry fares"
]

prompt_templates = [
    "Generate a question and detailed answer about {transport} {aspect} in Sri Lanka.",
    "What should a traveler know about {aspect} when using {transport} in Sri Lanka?",
    "Write a travel Q&A on {transport} and its {aspect}.",
    "Explain {aspect} for tourists planning to use {transport} in Sri Lanka.",
    "Create a Q&A about how to experience {transport}'s {aspect} in an authentic way.",
    "What makes {transport} in Sri Lanka special in terms of {aspect}?",
    "Describe a common tourist question and answer about {transport} related to {aspect}.",
    "How does Sri Lanka’s culture influence the experience of {aspect} in {transport}?",
    "Give a tourist guide Q&A focusing on {transport} and its {aspect}.",
    "Provide a Q&A about unique {aspect} found in {transport}, Sri Lanka."
]
air_prompt_templates = [
    "Generate a question and detailed answer about {aspect} for air travel from Sri Lanka.",
    "What should a traveler know about {aspect} when taking flights from Sri Lanka?",
    "Write a travel Q&A on air transportation and its {aspect} in Sri Lanka.",
    "Explain {aspect} for tourists planning to fly from Sri Lanka.",
    "Create a Q&A about {aspect} for international flights leaving Sri Lanka.",
    "Describe a common tourist question and answer about air travel from Sri Lanka related to {aspect}.",
    "How does Sri Lanka’s culture influence the experience of {aspect} in air travel?",
    "Give a tourist guide Q&A focusing on air transportation and its {aspect} from Sri Lanka.",
    "Provide a Q&A about unique {aspect} found in air travel from Sri Lanka."
]
ship_prompt_templates = [
    "Generate a question and detailed answer about {aspect} for ship travel from Sri Lanka.",
    "What should a traveler know about {aspect} when using ships or ferries from Sri Lanka?",
    "Write a travel Q&A on sea transportation and its {aspect} from Sri Lanka.",
    "Explain {aspect} for tourists planning to travel by ship from Sri Lanka.",
    "Create a Q&A about {aspect} for passengers on ships or ferries from Sri Lanka.",
    "Describe a common tourist question and answer about ship travel from Sri Lanka related to {aspect}.",
    "How does Sri Lanka’s maritime culture influence the experience of {aspect} in ship travel?",
    "Give a tourist guide Q&A focusing on sea transportation and its {aspect} from Sri Lanka.",
    "Provide a Q&A about unique {aspect} found in ship travel from Sri Lanka."
]

def generate_prompts(min_count=2100):
    prompts = []
    # Land: all combinations, all templates
    for t in transport_types:
        for a in transport_aspects:
            for template in prompt_templates:
                prompts.append(template.format(transport=t, aspect=a))
    # Air: all combinations, all templates
    for a in air_transport_aspects:
        for template in air_prompt_templates:
            prompts.append(template.format(aspect=a))
    # Ship: all combinations, all templates
    for a in ship_transport_aspects:
        for template in ship_prompt_templates:
            prompts.append(template.format(aspect=a))
    random.shuffle(prompts)
    # If somehow user asks for less, allow slicing; else ensure always 2100+
    return prompts[:max(min_count, 2100)]

def get_qa_from_llm(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a transportation expert on Sri Lanka. For each prompt, generate a clear question and detailed, informative answer about transportation in Sri Lanka. Format the response as a JSON with 'input' and 'output' keys."},
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
            except Exception:
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
            except Exception:
                pass
        elif response.status_code == 429:
            time.sleep(5)
        return None
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def clean_qa(qa):
    """Ensure input and output are strings and stripped, no nulls"""
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
    qa_pairs = []
    seen_inputs = set()
    if os.path.exists("sri_lanka_transportation_qa_temp.json"):
        with open("sri_lanka_transportation_qa_temp.json", "r", encoding="utf-8") as f:
            try:
                qa_pairs = json.load(f)
                seen_inputs = set(q["input"] for q in qa_pairs)
                print(f"Resuming from saved progress: {len(qa_pairs)} Q&A loaded.")
            except Exception:
                qa_pairs = []
                seen_inputs = set()
    prompts = generate_prompts(2100)
    print(f"Generating {len(prompts)} prompts to get at least 2000 unique transportation Q&A pairs...")

    for prompt in tqdm(prompts, desc="Generating Q&A pairs"):
        qa = get_qa_from_llm(prompt)
        if qa and "input" in qa and "output" in qa:
            qa = clean_qa(qa)
            if qa["input"] not in seen_inputs and qa["input"] and qa["output"]:
                qa_pairs.append(qa)
                seen_inputs.add(qa["input"])
                if len(qa_pairs) % 100 == 0:
                    with open("sri_lanka_transportation_qa_temp.json", "w", encoding="utf-8") as f:
                        json.dump(qa_pairs, f, ensure_ascii=False)
                    print(f"Saved {len(qa_pairs)} Q&A so far...")
        time.sleep(1)
        if len(qa_pairs) >= 2000:
            break
    print(f"\n✅ Completed: {len(qa_pairs)} Q&A pairs generated.")
    with open("sri_lanka_transportation_qa.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False)
    print("📁 Dataset saved to sri_lanka_transportation_qa.json")

if __name__ == "__main__":
    main()