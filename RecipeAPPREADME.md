# CSUSB-student-work

# Diet Rite App Reinforcement machine learning app built for self utilizing Bayseian Optomization GP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import ast

# ==== LOAD DATASETS ====
recipes = pd.read_csv('Receipes from around the world.csv', encoding='latin1')
allergens = pd.read_csv('food_ingredients_and_allergens.csv')
nutrition_tables = {
    'Atherosclerosis': pd.read_csv('Nutritional_Values_Applied_Diet_Atherosclerosis.csv'),
    'Hypertension': pd.read_csv('Nutritional_Values_Applied_Diet_Hypertension.csv'),
    'Type 2 Diabetes': pd.read_csv('Nutritional_Values_Applied_Diet_Type_2_Diabetes.csv')
}

print("\n=== Welcome to DietRiteAI! ===\n")
user_allergies = input("List your allergens (comma separated, e.g. dairy, peanuts, fish), or NONE: ").lower().replace(" ", "").split(",")
if user_allergies == ['none']: user_allergies = []

print("\nAvailable medical conditions:", ', '.join(nutrition_tables.keys()))
user_condition = input("Select your medical condition from above (or type NONE): ").strip()
if user_condition.lower() == "none": user_condition = None

# ---- Unique diets (unpack all list values) ----
def extract_all_diets(series):
    all_diets = set()
    for item in series.dropna():
        try:
            diets = ast.literal_eval(item)
            all_diets.update([d.lower() for d in diets])
        except:
            continue
    return sorted(all_diets)

all_diets = extract_all_diets(recipes['dietary_restrictions'])
all_cuisines = sorted(set(recipes['cuisine'].dropna().unique()))

print("\nDiet types found in recipes:", ', '.join(all_diets))
user_diet_input = input("Preferred diet types (comma separated, or type 'any' for all): ").strip().lower()
if user_diet_input == "any" or user_diet_input == "":
    user_diets = all_diets
else:
    user_diets = [d.strip() for d in user_diet_input.split(",") if d.strip()]

print("\nCuisine/taste types found:", ', '.join(all_cuisines))
user_cuisine = input("Preferred cuisine/taste (exact from list above, or leave blank for all): ").strip()
user_demo = input("\n(Optional) Any cultural/demographic preference (press Enter to skip): ").strip().lower()

cuisines = all_cuisines
diets = all_diets

def encode_recipe(row):
    features = []
    # Cuisine one-hot
    features += [1 if row['cuisine']==c else 0 for c in cuisines]
    # Diets: one-hot for each, present in row's restrictions
    try:
        recipe_diets = [d.lower() for d in ast.literal_eval(row['dietary_restrictions'])]
    except:
        recipe_diets = []
    features += [1 if d in recipe_diets else 0 for d in diets]
    # Normalized prep time
    prep_time = row['prep_time_minutes'] if not pd.isnull(row['prep_time_minutes']) else 30
    features.append(float(prep_time) / 120.0)
    return features

def encode_user(user_diets, user_cuisine):
    diet_feat = [1 if d in user_diets else 0 for d in diets]
    cuisine_feat = [1 if user_cuisine == c else 0 for c in cuisines]
    return diet_feat + cuisine_feat

user_vec = encode_user(user_diets, user_cuisine)

def is_recipe_safe(recipe_name, allergens_df, user_allergies):
    match = allergens_df[allergens_df['Food Product'].str.lower() == recipe_name.lower()]
    if match.empty: return True
    recipe_allergens = ','.join(match['Allergens'].astype(str)).lower()
    return not any(allergy in recipe_allergens for allergy in user_allergies)

def soft_filter_and_score(recipes, allergens, user_allergies, user_diets, user_cuisine, user_demo):
    results = []
    for _, row in recipes.iterrows():
        if not is_recipe_safe(row['recipe_name'], allergens, user_allergies): continue
        try:
            row_diets = [d.lower() for d in ast.literal_eval(row['dietary_restrictions'])]
        except:
            row_diets = []
        score = 0
        if user_diets and any(d in row_diets for d in user_diets): score += 1
        if user_cuisine and user_cuisine.lower() in str(row['cuisine']).lower(): score += 1
        if user_demo and user_demo in str(row['ingredients']).lower(): score += 1
        results.append((row, score))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return [r[0] for r in results if r[1]>0] or [r[0] for r in results][:10]

filtered_recipes = soft_filter_and_score(recipes, allergens, user_allergies, user_diets, user_cuisine, user_demo)

if not filtered_recipes:
    print("\nSorry, no recipes match your allergy filters. Try different options.\n")
    exit()

print("\n=== DietRiteAI: Adaptive Recipe Exploration (Real User Feedback) ===")
feature_vectors = []
for row in filtered_recipes:
    feature_vectors.append(encode_recipe(row) + user_vec)
X_candidates = np.array(feature_vectors)

n_rounds = min(10, len(X_candidates))  # Up to 10 suggestions
X_hist = []
y_hist = []

regret = []
eigval_records = []
mu_all, sigma_all = None, None

np.random.seed(42)
print("\nYou'll see up to 10 suggested recipes, and rate each one to help DietRiteAI learn:")

for t in range(n_rounds):
    if len(X_hist) < 2:
        idx = np.random.choice(len(X_candidates))
    else:
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True)
        gp.fit(np.array(X_hist), np.array(y_hist))
        mu, sigma = gp.predict(X_candidates, return_std=True)
        beta = 2.0
        ucb = mu + np.sqrt(beta) * sigma
        idx = np.argmax(ucb)

    X_hist.append(X_candidates[idx])
    row = filtered_recipes[idx]

    # Print ALL available info
    print(f"\n--- Recipe Suggestion #{t+1} ---")
    print(f"Recipe: {row['recipe_name']}")
    print(f"Cuisine: {row['cuisine']}")
    print(f"Diets: {row['dietary_restrictions']}")
    print(f"Ingredients: {row['ingredients']}")
    print(f"Cooking Time: {row['cooking_time_minutes']}")
    print(f"Prep Time: {row['prep_time_minutes']}")
    print(f"Servings: {row['servings']}")
    print(f"Calories per Serving: {row['calories_per_serving']}")
    print("-"*45)

    # User rates the recipe as reward (between 0 and 1)
    while True:
        try:
            reward = float(input("How much do you like this recipe? (rate 0 to 1): "))
            if 0 <= reward <= 1:
                break
            else:
                print("Please rate between 0 (worst) and 1 (best).")
        except Exception:
            print("Invalid input. Please enter a number between 0 and 1.")
    y_hist.append(reward)

    best_possible = 1.0
    regret.append(best_possible - reward)

    if len(X_hist) > 1:
        kernel = Matern(length_scale=1.0, nu=2.5)
        K = kernel(np.array(X_hist))
        eigvals = np.linalg.eigvalsh(K)
        eigval_records.append(eigvals)
    else:
        eigval_records.append(np.array([1.0]))

    if t == n_rounds - 1 and len(X_hist) > 1:
        gp.fit(np.array(X_hist), np.array(y_hist))
        mu_all, sigma_all = gp.predict(X_candidates, return_std=True)

    # Remove suggested recipe to avoid repeats
    X_candidates = np.delete(X_candidates, idx, axis=0)
    filtered_recipes.pop(idx)
    if len(X_candidates) == 0:
        break

print("\nDietRiteAI has finished its personalized suggestions!\n")

plt.figure(figsize=(6, 3))
plt.title("Cumulative Regret Over Time (GP-UCB)")
plt.plot(np.cumsum(regret), 'r-', lw=2)
plt.xlabel("Iteration")
plt.ylabel("Cumulative Regret")
plt.tight_layout()
plt.show()

min_eigs = [np.min(e) for e in eigval_records if len(e) > 0]
plt.figure(figsize=(6, 4))
plt.plot(min_eigs, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Minimum Eigenvalue')
plt.title('Minimum Eigenvalue over Time')
plt.grid(True)
plt.show()

if mu_all is not None and sigma_all is not None and len(mu_all) == len(X_candidates):
    plt.figure(figsize=(8, 5))
    plt.title("Final GP Estimate of Reward Function")
    plt.plot(range(len(mu_all)), mu_all, 'b-', label="GP Mean")
    plt.fill_between(range(len(mu_all)), mu_all-2*sigma_all, mu_all+2*sigma_all, color='b', alpha=0.2, label="95% Conf")
    plt.scatter(range(len(y_hist)), y_hist, c='k', s=40, alpha=0.7, label="Samples")
    plt.xlabel("Recipe Index (Filtered)")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
