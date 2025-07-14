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
    # bins are equal length so probabilities look like the normalizing frequencies
    probabilities = frequency / np.sum(frequency)
    bin_index = np.random.choice(len(frequency), p=probabilities)
    return np.random.uniform(binEdges[bin_index], binEdges[bin_index + 1])

def generate_variant_sample(input_sample, frequencies, binEdges, nVars, max_steps=1):
    """Generate a variant sample by sampling from the distribution of selected variables."""
    varsToShuffle = get_vars_shuffle(input_sample, nVars)
    adv = input_sample.copy()

    for _ in range(max_steps):
        for v in varsToShuffle:
            adv[v] = sample_with_frequencies(adv[v], frequencies[v], binEdges[v])
    return adv

if __name__ == "__main__":
    # Load dataset
    # df = pd.read_feather("../augmentation4/augmentation4.feather")
    df = pd.read_feather("../Val.feather")
    feature_columns = df.columns.drop("Label")
    X = df[feature_columns]
    y = df["Label"]

    # Parameters
    nBins = 200
    nVars = 10
    subset_size = 25000
    variants_per_sample = 50
    max_variants = 1250000

    # Compute frequencies and bin edges for each column
    frequencies, binEdges = get_frequencies(X, nBins)

    # Convert features to numpy array
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # Sample indices for augmentation
    subset_indices = np.random.choice(len(X_np), subset_size, replace=False)

    # Generate variant examples
    variant_examples = []
    for i in tqdm(subset_indices):
        for _ in range(variants_per_sample):
            adv = generate_variant_sample(X_np[i], frequencies, binEdges, nVars)
            variant_examples.append(np.append(adv, y_np[i]))
            if len(variant_examples) >= max_variants:
                break
        if len(variant_examples) >= max_variants:
            break

    # Save to feather
    if variant_examples:
        variant_df = pd.DataFrame(variant_examples, columns=list(feature_columns) + ["Label"])
        variant_df["Label"] = variant_df["Label"].astype(np.int32)
        print(f"Generated {len(variant_df)} variant samples (not verified as adversarial).")
        # variant_df.to_feather("variant_samples.feather")
        name = f"AugmentedvariantsfromVal__nbins_{nBins}__nvars_{nVars}__subsetsize_{subset_size}__variantspersample_{variants_per_sample}__maxvariants_{max_variants}"
        variant_df.to_feather(name)
    else:
        print("No variant samples generated.")

    # Verify output
df_out = pd.read_feather(name)
print(df_out.head(100))
print(len(df_out))
print(df_out["Label"].value_counts())