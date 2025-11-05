import streamlit as st
import pandas as pd
import random
import csv
import numpy as np

# --- PART A: THE GENETIC ALGORITHM "ENGINE" ---

def read_csv_to_dict(file_path):
    """
    Reads the new multi-column CSV and returns a dictionary 
    of {program: [final_rating]}.
    FIXED: Now uses .strip() to remove hidden spaces from data.
    """
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            try:
                # Read the header row
                header = next(reader)
                
                # Clean up header names (remove spaces, make lowercase)
                header_cleaned = [h.strip().lower() for h in header]

                # Find the column index for "Type of Program" and "Ratings"
                try:
                    program_col_index = header_cleaned.index('type of program')
                except ValueError:
                    st.error("Fatal Error: Could not find a 'Type of Program' column in your CSV.")
                    return {}
                
                try:
                    rating_col_index = header_cleaned.index('ratings')
                except ValueError:
                    st.error("Fatal Error: Could not find a 'Ratings' column in your CSV.")
                    return {}

            except StopIteration:
                st.error(f"Error: The file '{file_path}' is empty.")
                return {}
            
            # Read the data rows
            for row in reader:
                if len(row) > program_col_index and len(row) > rating_col_index:
                    
                    # --- THIS IS THE FIX ---
                    # Use .strip() to remove leading/trailing spaces
                    program = row[program_col_index].strip()
                    rating_str = row[rating_col_index].strip()
                    
                    if not program: # Skip empty program names
                        continue
                        
                    try:
                        # Fix typos like '9.8.0'
                        if rating_str.count('.') == 2:
                            parts = rating_str.split('.')
                            rating_str = f"{parts[0]}.{parts[1]}" # Re-assembles as '9.8'

                        # Get rating from the correct column index
                        rating = float(rating_str)
                        program_ratings[program] = [rating]  # Store as list for GA
                    except ValueError:
                        st.warning(f"Skipping row for '{program}': non-numeric rating value '{rating_str}'.")
    
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.info("Using sample data as a fallback.")
        # Fallback sample data (Final Ratings)
        program_ratings = {
            'documentary': [7.8], 'live_soccer': [8.0], 'music_program': [8.3],
            'news': [8.5], 'Boxing': [8.7], 'movie_a': [9.0],
            'reality_show': [9.2], 'movie_b': [9.3], 'tv_series_a': [9.7],
            'tv_series_b': [9.8]
        }
    return program_ratings

def fitness_function(schedule, ratings_data, schedule_length):
    """Calculates total fitness using modified single ratings per program."""
    total_rating = 0
    for program in schedule:
        if program in ratings_data:
            total_rating += ratings_data[program][0]  # Only one value per program
    return total_rating

def create_random_schedule(all_programs, schedule_length):
    """Creates a single, completely random schedule."""
    return [random.choice(all_programs) for _ in range(schedule_length)]

def crossover(schedule1, schedule2, schedule_length):
    """Performs single-point crossover."""
    if len(schedule1) < 2 or len(schedule2) < 2:
        return schedule1, schedule2
    crossover_point = random.randint(1, schedule_length - 1)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule, all_programs, schedule_length):
    """Mutates a schedule by changing one random program."""
    schedule_copy = schedule.copy()
    mutation_point = random.randint(0, schedule_length - 1)
    new_program = random.choice(all_programs)
    schedule_copy[mutation_point] = new_program
    return schedule_copy

def genetic_algorithm(ratings_data, all_programs, schedule_length,
                      generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, elitism_size=2):
    """Runs the genetic algorithm."""
    
    population = [create_random_schedule(all_programs, schedule_length) for _ in range(population_size)]
    
    best_schedule_ever = []
    best_fitness_ever = 0

    for generation in range(generations):
        pop_with_fitness = []
        for schedule in population:
            fitness = fitness_function(schedule, ratings_data, schedule_length)
            pop_with_fitness.append((schedule, fitness))
            
            if fitness > best_fitness_ever:
                best_fitness_ever = fitness
                best_schedule_ever = schedule

        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        new_population = []
        # Elitism
        for i in range(elitism_size):
            new_population.append(pop_with_fitness[i][0])

        # Fill the rest of the new population
        while len(new_population) < population_size:
            parent1 = random.choice(pop_with_fitness[:population_size // 2])[0]
            parent2 = random.choice(pop_with_fitness[:population_size // 2])[0]

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, schedule_length)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs, schedule_length)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs, schedule_length)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return best_schedule_ever, best_fitness_ever

# --- PART B: STREAMLIT INTERFACE ---

st.title("📺 Genetic Algorithm - TV Program Scheduling Optimizer")

# --- Load Data ---
file_path = 'program_ratings.modified.csv' 
ratings = read_csv_to_dict(file_path) # This now calls the new, fixed function

# Display the dataframe at the top
st.subheader("📊 Program Ratings Dataset")
try:
    # Load the full CSV from the path
    df_full = pd.read_csv(file_path)
    
    # But ONLY display the Program and its Final Rating for a clean UI
    df_display = df_full[['Type of Program', 'Ratings']] 
    st.dataframe(df_display)

except FileNotFoundError:
    st.info("Displaying fallback sample data instead.")
    df_display = pd.DataFrame({
        "Program": list(ratings.keys()),
        "Rating": [ratings[p][0] for p in ratings]
    })
    st.dataframe(df_display)
except (KeyError, pd.errors.ParserError):
    # This error happens if the columns aren't found or CSV is malformed
    st.error("Error: Your CSV must contain 'Type of Program' and 'Ratings' columns.")
    st.info("Displaying full CSV for debugging:")
    try:
        st.dataframe(df_full) # Show the full messy file so you can see the problem
    except:
        st.error("Could not even load the CSV file to display for debugging.")

if ratings:
    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))  # 6:00 to 23:00
    SCHEDULE_LENGTH = len(all_time_slots)  # 18 slots
    
    st.write(f"Successfully loaded {len(all_programs)} programs for optimization.")
    st.write(f"Schedule will be optimized for {SCHEDULE_LENGTH} time slots (6:00 to 23:00).")

    # --- Sidebar for GA Parameters ---
    st.sidebar.header("🧬 Set GA Parameters")

    # Trial 1
    st.sidebar.subheader("Trial 1")
    co_r_1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 0.95, 0.8, 0.05)
    mut_r_1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.01, 0.05, 0.02, 0.01)

    # Trial 2
    st.sidebar.subheader("Trial 2")
    co_r_2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 0.95, 0.9, 0.05)
    mut_r_2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.01, 0.05, 0.01, 0.01)

    # Trial 3
    st.sidebar.subheader("Trial 3")
    co_r_3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 0.95, 0.7, 0.05)
    mut_r_3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.01, 0.05, 0.05, 0.01)

    # --- Run Button ---
    if st.sidebar.button("🚀 Run All 3 Trials"):

        trials = [
            {"name": "Trial 1", "co": co_r_1, "mut": mut_r_1, "seed": 10},
            {"name": "Trial 2", "co": co_r_2, "mut": mut_r_2, "seed": 20},
            {"name": "Trial 3", "co": co_r_3, "mut": mut_r_3, "seed": 30},
        ]

        for t in trials:
            random.seed(t["seed"])
            np.random.seed(t["seed"])
            st.header(f"{t['name']} Results")
            st.write(f"*Parameters:* Crossover Rate = {t['co']}, Mutation Rate = {t['mut']}")
            schedule, fitness = genetic_algorithm(
                ratings_data=ratings,
                all_programs=all_programs,
                schedule_length=SCHEDULE_LENGTH,
                crossover_rate=t['co'],
                mutation_rate=t['mut']
            )
            
            # Get the rating for each program in the best schedule
            ratings_list = [ratings[program][0] for program in schedule]
            
            df_trial = pd.DataFrame({
                "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
                "Scheduled Program": schedule,
                "Rating": ratings_list  # <-- ADDED THIS COLUMN
            })

            st.dataframe(df_trial)
            st.write(f"*Best Fitness Score (Total Rating):* {fitness:.2f}")
            st.markdown("---")

else:
    st.error("Could not load any program data. Please check the file path and CSV content.")
