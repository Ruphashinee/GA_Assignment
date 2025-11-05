import streamlit as st
import pandas as pd
import random
import numpy as np
import csv

# --- PART A: THE GENETIC ALGORITHM "ENGINE" ---

def read_csv_to_dict(file_path):
    """Read CSV and return dictionary with final rating for GA and hourly ratings for display."""
    program_ratings = {}  # Final rating for GA
    hourly_ratings = {}   # Hourly ratings for display
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                program = row['Program']
                # Hourly ratings: Hour6 to Hour23
                hourly_ratings[program] = [float(row[f'Hour{i}']) for i in range(6,24)]
                # Final rating (column name is 'Ratings')
                program_ratings[program] = float(row['Ratings'])
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found. Using fallback sample data.")
        # Sample fallback data
        program_ratings = {
            'documentary': 7.8, 'live_soccer': 8.0, 'music_program': 8.3,
            'news': 8.5, 'Boxing': 8.7, 'movie_a': 9.0, 'reality_show': 9.2,
            'movie_b': 9.3, 'tv_series_a': 9.7, 'tv_series_b': 9.8
        }
        hourly_ratings = {p: [0.1]*18 for p in program_ratings}
    return program_ratings, hourly_ratings

def fitness_function(schedule, program_ratings, schedule_length):
    """Total fitness based on final ratings (1 value per program)."""
    total_rating = 0
    for program in schedule:
        if program in program_ratings:
            total_rating += program_ratings[program]
    average_rating = total_rating / schedule_length
    return total_rating, average_rating

def create_random_schedule(all_programs, schedule_length):
    return [random.choice(all_programs) for _ in range(schedule_length)]

def crossover(schedule1, schedule2, schedule_length):
    if len(schedule1) < 2 or len(schedule2) < 2:
        return schedule1, schedule2
    point = random.randint(1, schedule_length - 1)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2

def mutate(schedule, all_programs, schedule_length):
    schedule_copy = schedule.copy()
    point = random.randint(0, schedule_length - 1)
    schedule_copy[point] = random.choice(all_programs)
    return schedule_copy

def genetic_algorithm(program_ratings, all_programs, schedule_length,
                      generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, elitism_size=2):

    population = [create_random_schedule(all_programs, schedule_length) for _ in range(population_size)]
    best_schedule = []
    best_fitness = 0

    for _ in range(generations):
        pop_with_fitness = []
        for sched in population:
            fitness, _ = fitness_function(sched, program_ratings, schedule_length)
            pop_with_fitness.append((sched, fitness))
            if fitness > best_fitness:
                best_fitness = fitness
                best_schedule = sched

        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        new_pop = []

        # Elitism
        for i in range(elitism_size):
            new_pop.append(pop_with_fitness[i][0])

        # Fill rest of population
        while len(new_pop) < population_size:
            p1 = random.choice(pop_with_fitness[:population_size//2])[0]
            p2 = random.choice(pop_with_fitness[:population_size//2])[0]

            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2, schedule_length)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if random.random() < mutation_rate:
                c1 = mutate(c1, all_programs, schedule_length)
            if random.random() < mutation_rate:
                c2 = mutate(c2, all_programs, schedule_length)

            new_pop.append(c1)
            if len(new_pop) < population_size:
                new_pop.append(c2)

        population = new_pop

    return best_schedule, best_fitness

# --- PART B: STREAMLIT INTERFACE ---

st.title("📺 Genetic Algorithm - TV Program Scheduling Optimizer")

# Load CSV
file_path = 'Ratings.csv'  # <-- your file name
program_ratings, hourly_ratings = read_csv_to_dict(file_path)

# Display hourly table
st.subheader("📊 Hourly Ratings per Program")
df_hourly = pd.DataFrame.from_dict(hourly_ratings, orient='index')
df_hourly.columns = [f"Hour {i}" for i in range(6,24)]
df_hourly['Ratings'] = df_hourly.index.map(lambda x: program_ratings.get(x, 0))
st.dataframe(df_hourly)

all_programs = list(program_ratings.keys())
all_time_slots = list(range(6,24))
SCHEDULE_LENGTH = len(all_time_slots)

# Sidebar for GA parameters
st.sidebar.header("🧬 GA Parameters")

st.sidebar.subheader("Trial 1")
co_r1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 0.95, 0.8, 0.05)
mut_r1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.01, 0.05, 0.02, 0.01)

st.sidebar.subheader("Trial 2")
co_r2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 0.95, 0.9, 0.05)
mut_r2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.01, 0.05, 0.01, 0.01)

st.sidebar.subheader("Trial 3")
co_r3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 0.95, 0.7, 0.05)
mut_r3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.01, 0.05, 0.05, 0.01)

# Run GA
if st.sidebar.button("🚀 Run All Trials"):

    trials = [
        {"name": "Trial 1", "co": co_r1, "mut": mut_r1, "seed": 10},
        {"name": "Trial 2", "co": co_r2, "mut": mut_r2, "seed": 20},
        {"name": "Trial 3", "co": co_r3, "mut": mut_r3, "seed": 30}
    ]

    for t in trials:
        random.seed(t["seed"])
        np.random.seed(t["seed"])
        st.header(f"{t['name']} Results")
        st.write(f"*Parameters:* Crossover Rate = {t['co']}, Mutation Rate = {t['mut']}")
        schedule, total_fitness = genetic_algorithm(
            program_ratings=program_ratings,
            all_programs=all_programs,
            schedule_length=SCHEDULE_LENGTH,
            crossover_rate=t['co'],
            mutation_rate=t['mut']
        )
        _, avg_fitness = fitness_function(schedule, program_ratings, SCHEDULE_LENGTH)
        df_schedule = pd.DataFrame({
            "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
            "Scheduled Program": schedule
        })
        st.dataframe(df_schedule)
        st.write(f"*Total Fitness Score:* {total_fitness:.1f}")
        st.write(f"*Average Fitness per Slot:* {avg_fitness:.1f}")
        st.markdown("---")
