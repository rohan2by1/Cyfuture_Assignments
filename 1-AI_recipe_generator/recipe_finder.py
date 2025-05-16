import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

# Function to extract food entity (basic version)
def extract_food(user_input):
    doc = nlp(user_input)
    # Try to get noun chunks as possible dish names
    for chunk in doc.noun_chunks:
        return chunk.text
    return user_input  # fallback to full input

def find_recipe(food, dataset_path='recipes.csv'):
    df = pd.read_csv(dataset_path)
    # Try to find closest match by title
    matches = df[df['title'].str.contains(food, case=False, na=False)]
    if not matches.empty:
        recipe = matches.iloc[0]
        return f"""**{recipe['title']}**

**Ingredients:**
{recipe['ingredients']}

**Instructions:**
{recipe['instructions']}
"""
    else:
        return "Sorry, no recipe found for that food."