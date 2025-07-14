import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def get_vars_shuffle(input_sample, nVars):
    return random.sample(range(len(input_sample)), nVars)

def get_frequencies(data, nBins):
    frequencies = []
    binEdges = []
    for col in data.columns:
        freq, edges = np.histogram(data[col], bins=nBins, density=True)
        frequencies.append(freq)
        binEdges.append(edges)
    return frequencies, binEdges

def sample_with_frequencies(value, frequency, binEdges):
    probabilities = frequency / np.sum(frequency)
    bin_index = np.random.choice(len(frequency), p=probabilities)
    return np.random.uniform(binEdges[bin_index], binEdges[bin_index + 1])

def generate_variant_sample(input_sample, frequencies, binEdges, nVars, max_steps=1):
    varsToShuffle = get_vars_shuffle(input_sample, nVars)
    adv = input_sample.copy()
    for _ in range(max_steps):
        for v in varsToShuffle:
            adv[v] = sample_with_frequencies(adv[v], frequencies[v], binEdges[v])
    return adv

if __name__ == "__main__":
    df = pd.read_feather("../Train.feather")
    feature_columns = df.columns.drop("Label")
    X = df[feature_columns]
    y = df["Label"]

    # Parameters
    nBins = 100
    nVars = 10
    subset_size = 100000
    variants_per_sample = 50
    max_variants = 5000000
    chunk_size = 1000000  # Save every 1 million rows

    frequencies, binEdges = get_frequencies(X, nBins)
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    subset_indices = np.random.choice(len(X_np), subset_size, replace=False)

    chunk_index = 1
    current_chunk = []
    total_written = 0

    output_prefix = f"Augmentedvariants__nbins_{nBins}__nvars_{nVars}__subsetsize_{subset_size}__variantspersample_{variants_per_sample}__maxvariants_{max_variants}"

    for i in tqdm(subset_indices):
        for _ in range(variants_per_sample):
            adv = generate_variant_sample(X_np[i], frequencies, binEdges, nVars)
            current_chunk.append(np.append(adv, y_np[i]))
            if len(current_chunk) >= chunk_size:
                variant_df = pd.DataFrame(current_chunk, columns=list(feature_columns) + ["Label"])
                variant_df["Label"] = variant_df["Label"].astype(np.int32)
                part_name = f"{output_prefix}__part{chunk_index}.feather"
                variant_df.to_feather(part_name)
                print(f"Saved chunk {chunk_index} with {len(current_chunk)} rows.")
                total_written += len(current_chunk)
                chunk_index += 1
                current_chunk = []
            if total_written >= max_variants:
                break
        if total_written >= max_variants:
            break

    # Save any remaining samples
    if current_chunk:
        variant_df = pd.DataFrame(current_chunk, columns=list(feature_columns) + ["Label"])
        variant_df["Label"] = variant_df["Label"].astype(np.int32)
        part_name = f"{output_prefix}__part{chunk_index}.feather"
        variant_df.to_feather(part_name)
        print(f"Saved final chunk {chunk_index} with {len(current_chunk)} rows.")
        total_written += len(current_chunk)

    print(f"Total variant samples generated and saved: {total_written}")
