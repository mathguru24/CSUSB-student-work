# CSUSB-student-work

# Diet Rite App Reinforcement machine learning app built for self utilizing Bayseian Optomization GP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ==== LOAD DATASETS ====
recipes = pd.read_csv('data_cuisine[1].csv')
allergens = pd.read_csv('food_ingredients_and_allergens.csv')
nutrition_tables = {
    'Atherosclerosis': pd.read_csv('Nutritional_Values_Applied_Diet_Atherosclerosis.csv'),
    'Hypertension': pd.read_csv('Nutritional_Values_Applied_Diet_Hypertension.csv'),
    'Type 2 Diabetes': pd.read_csv('Nutritional_Values_Applied_Diet_Type_2_Diabetes.csv')
}

# ==== USER INPUT PROMPTING ====
print("\n=== Welcome to DietRiteAI! ===\n")
user_allergies = input("List your allergens (comma separated, e.g. dairy, peanuts, fish), or NONE: ").lower().replace(" ", "").split(",")
if user_allergies == ['none']: user_allergies = []

print("\nAvailable medical conditions:", ', '.join(nutrition_tables.keys()))
user_condition = input("Select your medical condition from above (or type NONE): ").strip()
if user_condition.lower() == "none": user_condition = None

all_diets = sorted(set(recipes['diet'].dropna().unique()))
all_cuisines = sorted(set(recipes['cuisine'].dropna().unique()))
print("\nDiet types found in recipes:", ', '.join(all_diets))
user_diet = input("Preferred diet type (exact from list above, e.g. Vegetarian): ").strip()
print("\nCuisine/taste types found:", ', '.join(all_cuisines))
user_cuisine = input("Preferred cuisine/taste (exact from list above, e.g. Indian): ").strip()
user_demo = input("\n(Optional) Any cultural/demographic preference (press Enter to skip): ").strip().lower()

def parse_prep_time(prep_time_str):
    try: return int(''.join(filter(str.isdigit, str(prep_time_str))))
    except: return 30  # default

cuisines = all_cuisines
courses = sorted(recipes['course'].dropna().unique())
diets = all_diets

def encode_recipe(row):
    features = []
    features += [1 if row['cuisine']==c else 0 for c in cuisines]
    features += [1 if row['course']==c else 0 for c in courses]
    features += [1 if row['diet']==c else 0 for c in diets]
    prep_time = parse_prep_time(row['prep_time'])
    features.append(prep_time / 120.0)
    return features

def encode_user(user_diet, user_cuisine):
    diet_feat = [1 if user_diet == d else 0 for d in diets]
    cuisine_feat = [1 if user_cuisine == c else 0 for c in cuisines]
    return diet_feat + cuisine_feat

user_vec = encode_user(user_diet, user_cuisine)

def is_recipe_safe(recipe_name, allergens_df, user_allergies):
    match = allergens_df[allergens_df['Food Product'].str.lower() == recipe_name.lower()]
    if match.empty: return True
    recipe_allergens = ','.join(match['Allergens'].astype(str)).lower()
    return not any(allergy in recipe_allergens for allergy in user_allergies)

def soft_filter_and_score(recipes, allergens, user_allergies, user_diet, user_cuisine, user_demo):
    results = []
    for _, row in recipes.iterrows():
        if not is_recipe_safe(row['name'], allergens, user_allergies): continue
        score = 0
        if user_diet and user_diet.lower() in str(row['diet']).lower(): score += 1
        if user_cuisine and user_cuisine.lower() in str(row['cuisine']).lower(): score += 1
        if user_demo and user_demo in str(row['description']).lower(): score += 1
        results.append((row, score))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return [r[0] for r in results if r[1]>0] or [r[0] for r in results][:10]  # fallback

filtered_recipes = soft_filter_and_score(recipes, allergens, user_allergies, user_diet, user_cuisine, user_demo)

if not filtered_recipes:
    print("\nSorry, no recipes match your allergy filters. Try different options.\n")
    exit()

# ==== GP-UCB LEARNING LOOP & METRICS ====
print("\n=== DietRiteAI: Adaptive Recipe Exploration (Real User Feedback) ===")
feature_vectors = []
for row in filtered_recipes:
    feature_vectors.append(encode_recipe(row) + user_vec)
X_candidates = np.array(feature_vectors)

n_rounds = min(5, len(X_candidates))  # Number of recipes to suggest
X_hist = []
y_hist = []

regret = []
eigval_records = []
mu_all, sigma_all = None, None

np.random.seed(42)
print("\nYou'll see up to 5 suggested recipes, and rate each one to help DietRiteAI learn:")

for t in range(n_rounds):
    # Initial picks: Random for the first 2
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

    # ---- NEW: REAL USER FEEDBACK ----
    row = filtered_recipes[idx]
    print(f"\n--- Recipe Suggestion #{t+1} ---")
    print(f"Recipe: {row['name']} | Cuisine: {row['cuisine']} | Diet: {row['diet']}")
    print(f"Description: {row['description'][:180]}...")
    print(f"Prep time: {row['prep_time']}")
    print(f"Instructions: {row['instructions'][:120]}...")
    print(f"Image: {row['image_url']}\n")
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

    # Regret (difference from max in this batch)
    best_possible = 1.0  # Since user can rate up to 1.0 (or use max(y_hist + [1.0]))
    regret.append(best_possible - reward)

    # Kernel eigenvalues
    if len(X_hist) > 1:
        kernel = Matern(length_scale=1.0, nu=2.5)
        K = kernel(np.array(X_hist))
        eigvals = np.linalg.eigvalsh(K)
        eigval_records.append(eigvals)
    else:
        eigval_records.append(np.array([1.0]))

    # Store final GP mean/std
    if t == n_rounds - 1 and len(X_hist) > 1:
        gp.fit(np.array(X_hist), np.array(y_hist))
        mu_all, sigma_all = gp.predict(X_candidates, return_std=True)

    # Remove suggested recipe to avoid repeats
    X_candidates = np.delete(X_candidates, idx, axis=0)
    filtered_recipes.pop(idx)
    if len(X_candidates) == 0:
        break

print("\nDietRiteAI has finished its personalized suggestions!\n")

# ==== DIETRITEAI METRICS & DIAGNOSTICS ====
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
