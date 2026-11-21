"""Create the two public Hugging Face collections for the submission."""
from huggingface_hub import HfApi
api = HfApi()
sets = {
    ("Courtier models", "LoRA adapters from the Courtier safety-drift study. Research only: less safe than their base models.", "model"):
        ["courtier-aya-expanse-8b-lora", "courtier-aya-expanse-32b-lora", "courtier-tiny-aya-global-lora"],
    ("Courtier datasets", "Training data, model answers, judge labels and result tables from the Courtier safety-drift study.", "dataset"):
        ["courtier-training-data", "courtier-results", "courtier-sycophancy-eval", "courtier-orbench-raw"],
}
for (title, desc, kind), names in sets.items():
    col = api.create_collection(title, namespace="Oakshat", description=desc, private=False, exists_ok=True)
    for n in names:
        api.add_collection_item(col.slug, item_id=f"Oakshat/{n}", item_type=kind, exists_ok=True)
    print(f"https://huggingface.co/collections/{col.slug}")
