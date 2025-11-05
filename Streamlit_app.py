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
                header = next(reader)  # Skip header
                if len(header) < 20:
                    st.warning(f"Warning: CSV file has {len(header)} columns. Expected at least 20.")
            except StopIteration:
                st.error(f"Error: The file '{file_path}' is empty.")
                return {}
            
            for row in reader:
                if len(row) >= 20:
                    program_name = row[0]
                    try:
                        hourly_ratings = [float(x) for x in row[1:19]]  # 18 slots
                        final_rating = float(row[19])
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
    Scaled ×10 so it matches the lecturer’s or friend’s score format (e.g., 84.6).
    """
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings_data:
            if time_slot < len(ratings_data[program]['hourly']):
                total_rating += ratings_data[program]['hourly'][time_slot]
    # Scale to match example format
    return total_rating * 10


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


# --- PART B: THE STREAMLIT WEB APPLICATION ---

st.title("📺 Genetic Algorithm - TV Program Scheduling Optimizer")

file_path = 'program_ratings.modified.csv'
ratings = read_csv_to_dict(file_path)

# Display the dataframe
st.subheader("📊 Program Ratings Dataset (from CSV)")
try:
    df_display = pd.read_csv(file_path)
    st.dataframe(df_display)
except FileNotFoundError:
    st.error(f"Could not find {file_path} to display.")


if ratings:
    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))  # 6:00 to 23:00
    SCHEDULE_LENGTH = len(all_time_slots)
    
    st.write(f"Successfully loaded {len(all_programs)} programs for optimization.")
    st.write(f"Schedule will be optimized for {SCHEDULE_LENGTH} time slots (6:00 to 23:00).")

    # --- Sidebar parameters ---
    st.sidebar.header("🧬 Set GA Parameters")

    st.sidebar.subheader("Trial 1")
    co_r_1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 0.95, 0.8, 0.05)
    mut_r_1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.01, 0.05, 0.02, 0.01)

    st.sidebar.subheader("Trial 2")
    co_r_2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 0.95, 0.95, 0.05)
    mut_r_2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.01, 0.05, 0.02, 0.01)

    st.sidebar.subheader("Trial 3")
    co_r_3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 0.95, 0.8, 0.05)
    mut_r_3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.01, 0.05, 0.05, 0.01)
    
    # --- Run button ---
    if st.sidebar.button("🚀 Run All 3 Trials"):

        def run_trial(seed, co_r, mut_r, trial_num):
            random.seed(seed)
            np.random.seed(seed)
            schedule, fitness = genetic_algorithm(
                ratings_data=ratings, all_programs=all_programs,
                schedule_length=SCHEDULE_LENGTH,
                crossover_rate=co_r, mutation_rate=mut_r
            )
            hourly_ratings = [ratings[p]['hourly'][i] for i, p in enumerate(schedule)]
            final_ratings = [ratings[p]['final'] for p in schedule]
            df = pd.DataFrame({
                "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
                "Scheduled Program": schedule,
                "Hourly Rating": hourly_ratings,
                "Final Rating": final_ratings
            })
            st.header(f"Trial {trial_num} Results")
            st.write(f"Parameters: Crossover Rate = {co_r}, Mutation Rate = {mut_r}")
            st.dataframe(df)
            st.write(f"*Best Fitness Score (Total Hourly Rating): {fitness:.1f}*")
            st.markdown("---")
            return fitness

        f1 = run_trial(10, co_r_1, mut_r_1, 1)
        f2 = run_trial(20, co_r_2, mut_r_2, 2)
        f3 = run_trial(30, co_r_3, mut_r_3, 3)

else:
    st.error("Could not load any program data. Please check the file path and CSV content.")
