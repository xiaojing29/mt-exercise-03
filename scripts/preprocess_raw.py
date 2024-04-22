import sys


def clean_line(line):
    # Split the line into components
    parts = line.split("+++$+++")
    if len(parts) >= 5:
        # Extract the dialogue part, which is the last element
        dialogue = parts[-1].strip()
        # Replace problematic characters and strip leading/trailing spaces
        dialogue = dialogue.replace('\ufeff', '').replace('\n', '').strip()
        # Normalize whitespace
        dialogue = " ".join(dialogue.split())
        return dialogue
    return None


def is_redundant(dialogue):
    """
    Checks if the dialogue is redundant or not very informative.
    """
    # Check for very short dialogues, which might not be informative.
    if len(dialogue.split()) < 3:
        return True

    # Check for repetitive words which might indicate less informative dialogue
    words = dialogue.split()
    if len(set(words)) < len(words) / 2:  # If less than half the words are unique
        return True

    return False


def main():
    lines = []
    for line in sys.stdin:
        cleaned_line = clean_line(line)
        if cleaned_line and not is_redundant(cleaned_line):
            lines.append(cleaned_line)

    # Take 1/5 of the dataset in the middle
    total_lines = len(lines)  # Total number of lines in the dataset
    start_index = total_lines * 40 // 100  # Start at 40% of the total
    end_index = start_index + total_lines * 20 // 100  # End after the next 20% of the total

    selected_lines = lines[start_index:end_index]  # Slice the middle one-fifth

    # Output the selected lines
    for selected_line in selected_lines:
        sys.stdout.write(selected_line + "\n")


if __name__ == "__main__":
    main()

