import warnings
# 1. Nuke the annoying Windows symlink warnings completely 
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import re
import csv
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelCard, login
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

# 2. Load the hidden token from your .env file
load_dotenv()
my_hf_token = os.getenv("HF_TOKEN")
if my_hf_token:
    login(my_hf_token)
else:
    print("WARNING: No HF_TOKEN found in .env file! Running unauthenticated.")

# --- CONFIGURATION ---
LIMIT = None  
OUTPUT_CSV = "huggingface_models.csv"
BASE_IMAGE_DIR = "images"
TRACKING_FILE = "processed_models.txt"

def setup_directories():
    if not os.path.exists(BASE_IMAGE_DIR):
        os.makedirs(BASE_IMAGE_DIR)

def load_processed_models():
    """Reads the tracking file and returns a set of already finished models."""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    return set()

def mark_as_processed(model_id):
    """Saves the model ID to the tracking file so we don't process it again."""
    with open(TRACKING_FILE, 'a', encoding='utf-8') as f:
        f.write(model_id + '\n')

def download_image(img_url, creator, model_name, image_index):
    # Remove characters that Windows folders hate (like * or <)
    safe_creator = "".join([c for c in creator if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()
    safe_model = "".join([c for c in model_name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()
    
    creator_dir = os.path.join(BASE_IMAGE_DIR, safe_creator)
    if not os.path.exists(creator_dir):
        os.makedirs(creator_dir, exist_ok=True)
    
    ext = os.path.splitext(img_url.split('?')[0])[-1].lower()
    if ext not in ['.png', '.jpg', '.jpeg']:
        ext = '.jpg'
        
    filename = f"{safe_model}_{image_index}{ext}"
    filepath = os.path.join(creator_dir, filename)
    
    try:
        # Handle relative URLs by prepending the Hugging Face base URL
        if img_url.startswith('/'):
            img_url = f"https://huggingface.co{img_url}"
            
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception:
        pass # Silently skip broken image links
    return False

def extract_images_from_markdown(markdown_text):
    valid_images = []
    
    # Regex to catch Markdown and HTML images
    md_matches = re.finditer(r'!\[(.*?)\]\((.*?)\)', markdown_text)
    html_matches = re.finditer(r'<img[^>]+src=["\'](.*?)["\'][^>]*>', markdown_text)
    
    # Keywords to filter out logos, badges, and buttons
    junk_keywords = [
        'shields.io', 'badge', 'discord', 'logo', '.svg', 'avatar', 'icon',
        'colab', 'github', 'twitter', 'youtube', 'ko-fi', 'buymeacoffee',
        'sponsor', 'donate', 'button', 'social'
    ]
    
    for match in md_matches:
        alt_text = match.group(1).lower()
        img_url = match.group(2)
        combined_text = alt_text + " " + img_url.lower()
        if not any(keyword in combined_text for keyword in junk_keywords):
            clean_url = img_url.split(' ')[0].strip() 
            valid_images.append(clean_url)
            
    for match in html_matches:
        full_tag = match.group(0).lower()
        img_url = match.group(1)
        if not any(keyword in full_tag for keyword in junk_keywords):
            valid_images.append(img_url.strip())
            
    # Remove duplicates but keep order
    return list(dict.fromkeys(valid_images))

def clean_model_card(text):
    if not text:
        return "No Model Card"
    
    # Remove Markdown image tags
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove HTML tags (like <img>, <div>)
    text = re.sub(r'<[^>]+>', '', text)
    # Replace line breaks with a space to keep it contained in one CSV cell
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def main():
    setup_directories()
    api = HfApi()
    
    # Load our progress to prevent duplicates
    processed_models = load_processed_models()
    if processed_models:
        print(f"Resuming... Found {len(processed_models)} previously finished models. Skipping those!")
    
    print("Fetching the list of TEXT GENERATION models from Hugging Face...")
    
    # THE ONLY CHANGE IS HERE: We added filter="text-generation" 
    models = api.list_models(filter="text-generation", limit=LIMIT, fetch_config=True)
    
    file_exists = os.path.isfile(OUTPUT_CSV)
    
    # Open CSV in append mode ('a') so we add to it continually
    with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['task', 'creator', 'model_name', 'url', 'model_card', 'image_count']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
        if not file_exists:
            writer.writeheader()
    
        count = 0
        skipped = 0
        for model in models:
            # Check if we already processed this model
            if model.id in processed_models:
                skipped += 1
                if skipped % 5000 == 0:
                    print(f"Skipped {skipped} already-processed models...")
                continue
                
            try:
                model_id = model.id
                
                if '/' in model_id:
                    creator, model_name = model_id.split('/', 1)
                else:
                    creator, model_name = "huggingface", model_id
                    
                url = f"https://huggingface.co/{model_id}"
                tasks = [model.pipeline_tag] if model.pipeline_tag else ["No Task Specified"]
                
                model_card_text = "No Model Card"
                image_urls = []
                
                # Fetch Model Card Text
                try:
                    card = ModelCard.load(model_id)
                    raw_text = card.text
                    image_urls = extract_images_from_markdown(raw_text)
                    model_card_text = clean_model_card(raw_text) 
                except (RepositoryNotFoundError, EntryNotFoundError):
                    pass 
                except Exception:
                    pass
                    
                # Download Images
                image_count = 0
                for idx, img_url in enumerate(image_urls, start=1):
                    if download_image(img_url, creator, model_name, idx):
                        image_count += 1
                        
                # Write to CSV
                for task in tasks:
                    writer.writerow({
                        "task": task,
                        "creator": creator,
                        "model_name": model_name,
                        "url": url,
                        "model_card": model_card_text,
                        "image_count": image_count
                    })
                
                # Instantly save data to hard drive
                csv_file.flush() 
                
                # Log success so we don't repeat it next time
                mark_as_processed(model_id)
                processed_models.add(model_id) 
                
                count += 1
                print(f"Processed #{count}: {model_id} | Images: {image_count}")
                
            except Exception as e:
                print(f"Error processing {model.id}: {e}. Skipping to next...")

    print(f"\nFinished scraping! All data safely stored in {OUTPUT_CSV}")

if __name__ == "__main__":
    main()