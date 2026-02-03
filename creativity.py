# Imports

import pandas as pd
import json
import torch
import nltk
import spacy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NLTK_DATA'] = '/scratch.hpc/alessandro.tutone/nltk_data'
nltk.data.path.clear()  # clear default home paths
nltk.data.path.append('/scratch.hpc/alessandro.tutone/nltk_data')

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from diversity import compression_ratio, extract_patterns
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig#, infer_device
from IPython.display import display, Markdown
from creativity_index.DJ_search_exact import dj_search

nltk.download("punkt_tab", download_dir="/scratch.hpc/alessandro.tutone/nltk_data")
nltk.download("averaged_perceptron_tagger_eng", download_dir="/scratch.hpc/alessandro.tutone/nltk_data")

HF_TOKEN = 'hf_SULLXJJXoqqHXo' +'LBqyjXTdOkwjybapbPGF'


# Dataset preparation

ds = load_dataset("euclaise/writingprompts")
df = pd.DataFrame(ds['train'])
df_wp = df[df['prompt'].str.startswith('[ WP ] ')].copy()
df_wp['prompt'] = df_wp['prompt'].str.slice(7)
dataset = df_wp

current_path = '/scratch.hpc/alessandro.tutone/LLM_creativity'
os.makedirs(current_path + '/creativity_index/data/writingprompts/', exist_ok=True)
dataset_dict = [{"prompt": row.prompt, "text": row.story} for idx, row in dataset.iterrows()]
with open(current_path + "/creativity_index/data/writingprompts/dataset.json", "w") as final:
    json.dump(dataset_dict, final, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else str(x))


# Creatiivty Index

def compute_creativity_index(data_path, output_dir, subset=1, lm_tokenizer=False):
    os.makedirs(output_dir, exist_ok=True)
    print("\tRunning DJ Search for L=5 to L=12...")
    ngram_range = range(5, 13)
    
    for min_ngram in ngram_range:
        output_file = os.path.join(output_dir, f'L_{min_ngram}.json')
        dj_search(data_path, output_file, min_ngram=min_ngram, subset=subset, lm_tokenizer=lm_tokenizer)
    
    print("\tLoading files...")
    values = {}

    for min_ngram in ngram_range:
        output_file = os.path.join(output_dir, f'L_{min_ngram}.json')
        with open(output_file, 'r') as f:
            values[min_ngram] = json.load(f)

    print("\tComputing CI values...")
    creativity_index_values = []

    for text_idx in tqdm(range(subset), desc = '\tCreativity Index'):
        creativity_index = sum( 1 - values[min_ngram][text_idx]['coverage'] for min_ngram in ngram_range )
        creativity_index_values.append(creativity_index)

    return creativity_index_values


# Perplexity 

def perplexity(dataset_path, subset=1, model=False, tokenizer=False, model_name='gpt2'):
  if not model or not tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, device_map='auto')
  device = model.device

  with open(dataset_path, 'r') as f:
    dataset = json.load(f)

  perplexities = []

  for i in tqdm(range(subset), desc='\tPerplexity'):
    inputs = tokenizer(dataset[i]['text'], return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
      outputs = model(**inputs, labels=inputs["input_ids"])
      loss = outputs.loss
      perplexity = torch.exp(loss)
      perplexities.append(perplexity.item())

  return perplexities


# Syntactic templates
## CR-POS

def cr_pos(dataset_path, subset=1):
  with open(dataset_path, 'r') as f:
    dataset = json.load(f)

  if subset <= 0:
    return []

  subset = min(subset, len(dataset))
  cr_poses = []

  for text_idx in tqdm(range(subset), desc='\tCR-POS'):
    words = word_tokenize(dataset[text_idx]['text'])
    pos_tags = pos_tag(words)
    pos_tag_list = [pos_tag[1] for pos_tag in pos_tags]

    if not pos_tag_list:
      cr_poses.append(0)
      continue

    tags_sequence = " ".join(pos_tag_list)
    cr = compression_ratio(tags_sequence, algorithm='gzip', verbose = False)
    cr_poses.append(cr)

  return cr_poses

## Template Rate

# original paper uses templates of length n ∈ {4, 5, 6, 7, 8}
def template_rate(dataset_path, subset=1, len_template=4, top_n_templates=100):
  with open(dataset_path, 'r') as f:
    dataset = json.load(f)

  if subset <= 0:
    return []

  subset = min(subset, len(dataset))
  template_rates = []

  for text_idx in tqdm(range(subset), desc='\tTemplate Rate'):
    words = word_tokenize(dataset[text_idx]['text'])
    pos_tags = pos_tag(words)
    pos_tag_list = [pos_tag[1] for pos_tag in pos_tags]

    if not pos_tag_list:
      template_rates.append(0)
      continue

    patterns = extract_patterns([dataset[text_idx]['text']], n=len_template, top_n=top_n_templates)
    templates = set(patterns.keys())
    mask = [False] * len(pos_tag_list)

    for i in range(len(pos_tag_list)-len_template+1):
      if " ".join(pos_tag_list[i:i+len_template]) in templates:
        mask[i:i+len_template] = [True]*len_template

    template_rate = sum(mask)/len(pos_tag_list)
    template_rates.append(template_rate)

  return template_rates

## Template-per-Token

def template_per_token(dataset_path, subset=1, len_template=4, top_n_templates=1):
  with open(dataset_path, 'r') as f:
    dataset = json.load(f)

  if subset <= 0:
    return []

  subset = min(subset, len(dataset))
  tpts = []

  for text_idx in tqdm(range(subset), desc='\tTemplate-per-Token'):
    words = word_tokenize(dataset[text_idx]['text'])
    pos_tags = pos_tag(words)
    pos_tag_list = [pos_tag[1] for pos_tag in pos_tags]

    if not pos_tag_list:
      tpts.append(0)
      continue

    patterns = extract_patterns([dataset[text_idx]['text']], n=len_template, top_n=top_n_templates)
    templates = set(patterns.keys())
    num_templates_per_token = [0] * len(pos_tag_list)

    for i in range(len(pos_tag_list) - len_template + 1):
      template = " ".join(pos_tag_list[i:i+len_template])
      if template in templates:
        for j in range(i, i + len_template):
          num_templates_per_token[j] += 1

    num_words = len(words)
    tpt = sum(num_templates_per_token) / max(1,num_words)
    tpts.append(tpt)

  return tpts


# LLM-as-a-judge

def prepare_prompts(texts, prompt_template, tokenizer, metrics=[], generation_prompt = True):
  texts_formatted = []
  chat_template = tokenizer.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=generation_prompt)

  for text in texts:
    for metric in metrics:
      text_formatted = chat_template.format(text=text, metric=metric)
      texts_formatted.append(text_formatted)
    if not metrics:
      text_formatted = chat_template.format(text=text)
      texts_formatted.append(text_formatted)

  return texts_formatted

def fix_json_brackets(s):
    s = s.strip()
    open_braces = s.count('{')
    close_braces = s.count('}')
    
    while close_braces > open_braces and s.endswith('}'):
        s = s[:-1]
        close_braces -= 1
    
    return s

def generate_responses_batched(model, prompts: list, tokenizer, batch_size=8, **gen_param):
    responses = []
    device = model.device

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding='longest', padding_side='left', truncation=True).to(device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_param)

        input_lenght = inputs["input_ids"].shape[1]
        responses_tokens = outputs[:, input_lenght:]
        batch_responses = tokenizer.batch_decode(responses_tokens, skip_special_tokens=True)

        for response in batch_responses:
          try:
            response = fix_json_brackets(response)
            response_dictionary = json.loads(response)
            responses.append(response_dictionary)
          except json.JSONDecodeError:
            print(f"Problems while decoding the response:\n {response}")
            responses.append(None)
          except:
            print("Other problems occurred!")

    return responses

def llm_as_a_judge(model, tokenizer, chat_prompt, dataset_path, metrics=[], subset=1, batch_size=8, gen_params={}):
  with open(dataset_path, 'r') as f:
    dataset = json.load(f)

  if subset <= 0:
    return []

  subset = min(subset, len(dataset))
  texts = [dataset[i]['text'] for i in range(subset)]

  formatted_chat_prompts = prepare_prompts(texts, chat_prompt, tokenizer, metrics, generation_prompt = True)
  responses = generate_responses_batched(model, formatted_chat_prompts, tokenizer, batch_size=batch_size, **gen_params)

  return responses


# Creativity evaluation - All metrics together

def creativity_evaluation(model, tokenizer, chat_prompt, dataset_path, output_path, metrics=[], subset=2, generation_params={}, batch_size=8):

  print('STARTING WITH THE EVALUATION...')

  print('Computing Creativity Index...')
  creativity_index_values = compute_creativity_index(dataset_path, output_dir = current_path + '/creativity_index/outputs/writingprompts/L/', subset=subset, lm_tokenizer=False)
  print('Computing Perplexity...')
  perplexities = perplexity(dataset_path, subset=subset, model=model, tokenizer=tokenizer)
  print('Computing CR-POS...')
  cr_poses = cr_pos(dataset_path, subset=subset)
  print('Computing Template Rate...')
  template_rates = template_rate(dataset_path, subset=subset, len_template=4, top_n_templates=100)
  print('Computing Template-per-Token...')
  tpts = template_per_token(dataset_path, subset=subset, len_template=4, top_n_templates=100)
  print('Computing LLM-as-a-judge...')
  responses = llm_as_a_judge(model, tokenizer, chat_prompt, dataset_path, metrics, subset=subset, batch_size=batch_size, gen_params=generation_params)
  print('DONE!')

  creativity_metrics = {
      'creativity_index': creativity_index_values,
      'perplexity': perplexities,
      'cr_pos': cr_poses,
      'template_rate': template_rates,
      'template_per_token': tpts,
      'llm_as_a_judge': responses
  }

  os.makedirs(output_path, exist_ok=True)
  output_file = os.path.join(output_path, 'creativity_metrics.json')

  with open(output_file, 'w') as fp:
    json.dump(creativity_metrics, fp)

  return creativity_metrics


# RUNNING params

dataset_path = current_path + '/creativity_index/data/writingprompts/dataset.json'
output_path = current_path + '/results/'

# Parameters
subset = 10
batch_size = 1
model_name = "meta-llama/Llama-3.3-70B-Instruct"
metrics = ["surprise", "novelty", "value", "authenticity", "originality", "effectiveness", "fluency", "flexibility", "elaboration", "usefulness", "creativity"]

generation_params = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0,
    "top_p": 1,
    "repetition_penalty": 1.2
}

# Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_use_double_quant=True, # this further reduces the precision of weights (double quantization)
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    quantization_config=bnb_config,
    token=HF_TOKEN,
    device_map='auto'
)

chat_prompt_metric = [
    {
        'role': 'system',
        'content': (
          'You are an objective text evaluator. Your task is to assess a given text objectively according to a specific metric. '
          'Your only output must be a single, syntactically correct JSON object. '
          'Never include explanations, reasoning, or extra text outside the JSON. '

        )
    },
    {
        'role': 'user',
        'content': """Given the following text, you need to evaluate it for the specified metric: {metric}.
        
        ### Task
        Taking into account the definition of the specified metric, produce:
            - score: integer 1–5 (1 = lowest, 5 = highest)
            - justification: ≤30 words explaining the rating
            - excerpt: ≤20 words from the text supporting the evaluation (or null if not applicable)

        ### Rules
        Strict rules:
          • Evaluate the aspect objectively.
          • Do NOT reveal chain-of-thought. Provide only the requested justifications and evidence.
          • If the text is ambiguous or too short to be judged, score 3 and note "insufficient evidence".
          • Return **only valid JSON** with field: "name of the metric". The field must be an object with keys: score (int), justification (string), excerpt (string or null).
          • Do NOT answer anything else other than the JSON.
          • Do NOT include backticks, markdown, explanations, or anything outside the JSON.
          • If unsure about JSON syntax, default to minimal valid JSON with null excerpt.
          • Before answering, double check the brackets and the correctenss of the JSON

        Output structure:
            {{
              "{metric}":{{
              "score": <int>, 
              "justification": "<string>",
              "excerpt": "<string or null>"
              }}
            }}

        SCALE ANCHORS (use these as guidance):
          • 5 = clear, strong, unambiguous evidence for the aspect.
          • 4 = good evidence, minor weaknesses.
          • 3 = ambiguous or mixed evidence; could go either way.
          • 2 = weak evidence or some counter-evidence.
          • 1 = no evidence or direct counter-evidence.

        INPUT:
        Text to evaluate:
        "{text}"

        OUTPUT:
        - JSON object (as described).
        - The JSON must start with '_curly bracket_' and end with exactly one '_curly bracket_'.
        - Ensure the correctness of the JSON. Double check that the brackets are correct.

        End.
        """
    }
]

creativity_metrics = creativity_evaluation(model, tokenizer, chat_prompt_metric, dataset_path, output_path, metrics, subset=subset, generation_params=generation_params, batch_size=batch_size)
