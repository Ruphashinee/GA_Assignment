import streamlit as st
import pandas as pd
import random
import csv
import numpy as np

# --- PART A: THE GENETIC ALGORITHM "ENGINE" ---

def read_csv_to_dict(file_path):
    """
    Reads the CSV file.
    Assumes CSV structure: Program, hr1, hr2, ... hr18, FinalRating
    (20 columns total: 1 program + 18 hourly + 1 final)
    """
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            try:
                header = next(reader) # Skip header
                # Check for expected number of columns
                if len(header) < 20:
                    st.warning(f"Warning: CSV file has {len(header)} columns. Expected at least 20.")
            except StopIteration:
                st.error(f"Error: The file '{file_path}' is empty.")
                return {}
            
            for row in reader:
                if len(row) >= 20: # Ensure row has enough data
                    program_name = row[0]
                    try:
                        # Hourly ratings are columns 1 through 18
                        hourly_ratings = [float(x) for x in row[1:19]] # 18 slots
                        
                        # Final rating is column 19
                        final_rating = float(row[19])
                        
                        # Store in the new, required structure
                        program_ratings[program_name] = {
                            'hourly': hourly_ratings,
                            'final': final_rating
                        }
                    except ValueError:
                        st.warning(f"Skipping row for '{program_name}': non-numeric rating.")
    
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.info("Please make sure the file is in your GitHub repository.")
        return {}
    
    return program_ratings


def fitness_function(schedule, ratings_data, schedule_length):
    """
    Calculates the total fitness (sum of HOURLY ratings) of a given schedule.
    This is what the GA tries to MAXIMIZE.
    """
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings_data:
            # Use the 'hourly' list for the fitness calculation
            if time_slot < len(ratings_data[program]['hourly']):
                total_rating += ratings_data[program]['hourly'][time_slot]
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
    """Mutates a schedule by changing one random gene."""
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
    best_fitness_ever = 0 # This will be the highest TOTAL hourly rating

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


# --- PART B: THE STREAMLIT WEB APPLICATION ---

st.title("📺 Genetic Algorithm - TV Program Scheduling Optimizer")

# --- 1. Load Data ---
file_path = 'program_ratings_modified.csv' 
ratings = read_csv_to_dict(file_path)

# Display the dataframe at the top
st.subheader("📊 Program Ratings Dataset (from CSV)")
try:
    df_display = pd.read_csv(file_path)
    st.dataframe(df_display)
except FileNotFoundError:
    st.error(f"Could not find {file_path} to display.")


if ratings:
    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))  # 6:00 to 23:00
    SCHEDULE_LENGTH = len(all_time_slots)  # 18 slots
    
    st.write(f"Successfully loaded {len(all_programs)} programs for optimization.")
    st.write(f"Schedule will be optimized for {SCHEDULE_LENGTH} time slots (6:00 to 23:00).")

    # --- 2. Sidebar for Parameter Input ---
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

    # --- 3. Run Button and Display Results ---
    if st.sidebar.button("🚀 Run All 3 Trials"):

        # --- TRIAL 1 ---
        random.seed(10)
        np.random.seed(10)
        st.header("Trial 1 Results")
        st.write(f"Parameters: Crossover Rate = {co_r_1}, Mutation Rate = {mut_r_1}")
        schedule_1, fitness_1 = genetic_algorithm(
            ratings_data=ratings, all_programs=all_programs, schedule_length=SCHEDULE_LENGTH,
            crossover_rate=co_r_1, mutation_rate=mut_r_1
        )
        
        # --- NEW: Build the detailed DataFrame ---
        hourly_ratings_1 = []
        final_ratings_1 = []
        for time_slot, program in enumerate(schedule_1):
            hourly_ratings_1.append(ratings[program]['hourly'][time_slot])
            final_ratings_1.append(ratings[program]['final'])

        df_1 = pd.DataFrame({
            "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
            "Scheduled Program": schedule_1,
            "Hourly Rating": hourly_ratings_1,
            "Final Rating": final_ratings_1
        })
        st.dataframe(df_1)
        st.write(f"**Best Fitness Score (Total Hourly Rating): {fitness_1:.2f}**")
        st.markdown("---")


        # --- TRIAL 2 ---
        random.seed(20)
        np.random.seed(20)
        st.header("Trial 2 Results")
        st.write(f"Parameters: Crossover Rate = {co_r_2}, Mutation Rate = {mut_r_2}")
        schedule_2, fitness_2 = genetic_algorithm(
            ratings_data=ratings, all_programs=all_programs, schedule_length=SCHEDULE_LENGTH,
            crossover_rate=co_r_2, mutation_rate=mut_r_2
        )
        
        # --- NEW: Build the detailed DataFrame ---
        hourly_ratings_2 = []
        final_ratings_2 = []
        for time_slot, program in enumerate(schedule_2):
            hourly_ratings_2.append(ratings[program]['hourly'][time_slot])
            final_ratings_2.append(ratings[program]['final'])
            
        df_2 = pd.DataFrame({
            "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
            "Scheduled Program": schedule_2,
            "Hourly Rating": hourly_ratings_2,
            "Final Rating": final_ratings_2
        })
        st.dataframe(df_2)
        st.write(f"**Best Fitness Score (Total Hourly Rating): {fitness_2:.2f}**")
        st.markdown("---")


        # --- TRIAL 3 ---
        random.seed(30)
        np.random.seed(30)
        st.header("Trial 3 Results")
        st.write(f"Parameters: Crossover Rate = {co_r_3}, Mutation Rate = {mut_r_3}")
        schedule_3, fitness_3 = genetic_algorithm(
            ratings_data=ratings, all_programs=all_programs, schedule_length=SCHEDULE_LENGTH,
            crossover_rate=co_r_3, mutation_rate=mut_r_3
        )
        
        # --- NEW: Build the detailed DataFrame ---
        hourly_ratings_3 = []
        final_ratings_3 = []
        for time_slot, program in enumerate(schedule_3):
            hourly_ratings_3.append(ratings[program]['hourly'][time_slot])
            final_ratings_3.append(ratings[program]['final'])

        df_3 = pd.DataFrame({
            "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
            "Scheduled Program": schedule_3,
            "Hourly Rating": hourly_ratings_3,
            "Final Rating": final_ratings_3
        })
        st.dataframe(df_3)
        st.write(f"**Best Fitness Score (Total Hourly Rating): {fitness_3:.2f}**")
        st.markdown("---")

else:
    st.error("Could not load any program data. Please check the file path and CSV content.")
