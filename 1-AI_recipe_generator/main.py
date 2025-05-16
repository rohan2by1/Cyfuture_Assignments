import streamlit as st
from recipe_finder import find_recipe, extract_food

st.title("🍳 AI Recipe Generator")

user_input = st.text_input("What food or dish do you want a recipe for? (e.g. apple pie, chocolate cake)")

if user_input:
    dish = extract_food(user_input)
    result = find_recipe(dish)
    st.markdown(result)