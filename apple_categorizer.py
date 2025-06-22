import csv
import math
from collections import Counter


# Load CSV data
def load_data(filename):
    data = []
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            size = int(row["size"])
            texture = int(row["texture"])
            category = row["category"]
            data.append((size, texture, category))
    return data


# Calculate Euclidean distance
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Predict category using KNN
def predict_category(data, new_item, k=5):
    distances = []
    for item in data:
        dist = calculate_distance((item[0], item[1]), new_item)
        distances.append((dist, item[2]))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get top-k neighbors
    nearest = distances[:k]
    categories = [category for _, category in nearest]

    # Majority vote
    most_common = Counter(categories).most_common(1)[0][0]
    return most_common


# --- Main Program ---
if __name__ == "__main__":
    dataset = load_data("apple_data.csv")

    # Take input
    size = int(input("Enter apple size (e.g., 1 to 5): "))
    texture = int(input("Enter apple texture (e.g., 1 to 9): "))
    test_apple = (size, texture)

    result = predict_category(dataset, test_apple, k=5)
    print("Predicted Category:", result)
