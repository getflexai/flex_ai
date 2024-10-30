import matplotlib.pyplot as plt

def plot_token_length_distribution(token_lengths, title):
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def generate_report(max_seq_len_train, max_seq_len_eval):
    report = "Dataset is Valid for training:\n"
    report += "===================\n"
    report += f"Max sequence length in train dataset: {max_seq_len_train}\n"
    if max_seq_len_eval is not None:
        report += f"Max sequence length in eval dataset: {max_seq_len_eval}\n"
    else:
        report += "No evaluation dataset provided.\n"

    if max_seq_len_eval and max_seq_len_eval > max_seq_len_train:
        report += "Warning: The evaluation dataset has a longer sequence than the training dataset.\n"
    report += "\nConsiderations:\n"
    report += "- Check if a single example is causing a very high max token length.\n"
    report += "- Adjust context size if necessary.\n"

    return report