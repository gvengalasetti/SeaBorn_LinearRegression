import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

mean_set_A = np.random.multivariate_normal([1, 0], np.eye(2), 5)  # Cluster 1 means
mean_set_B = np.random.multivariate_normal([0, 1], np.eye(2), 5)  # Cluster 2 means

mean_collection = np.vstack((mean_set_A, mean_set_B))

data_points = []
point_labels = []

for i in range(100):
    chosen_mean = mean_collection[np.random.choice(len(mean_collection))]
    new_point = np.random.multivariate_normal(chosen_mean, np.eye(2) / 5)
    data_points.append(new_point)
    if np.any(np.isclose(chosen_mean, mean_set_A).all(axis=1)):
        point_labels.append(1)  # Label for set A
    else:
        point_labels.append(0)  # Label for set B

data_points = np.array(data_points)
point_labels = np.array(point_labels)

plt.scatter(data_points[:, 0], data_points[:, 1], c=point_labels, cmap='coolwarm', alpha=0.6)
plt.title('Simulated Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

X_input = data_points
y_target = point_labels

linear_model = LinearRegression()
linear_model.fit(X_input, y_target)

x_bounds = np.linspace(X_input[:, 0].min(), X_input[:, 0].max(), 200)
y_bounds = np.linspace(X_input[:, 1].min(), X_input[:, 1].max(), 200)
grid_X, grid_Y = np.meshgrid(x_bounds, y_bounds)

boundary_prediction = linear_model.predict(np.c_[grid_X.ravel(), grid_Y.ravel()]).reshape(grid_X.shape)

plt.contourf(grid_X, grid_Y, boundary_prediction, levels=[-0.5, 0.5, 1], cmap='coolwarm', alpha=0.3)
plt.scatter(X_input[:, 0], X_input[:, 1], c=y_target, cmap='coolwarm', edgecolor='black')
plt.title('Least Squares Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

predicted_labels = linear_model.predict(X_input) >= 0.5

false_positive_rate = np.sum((y_target == 0) & (predicted_labels == 1)) / np.sum(y_target == 0)
false_negative_rate = np.sum((y_target == 1) & (predicted_labels == 0)) / np.sum(y_target == 1)

print(f"False Positive Rate: {false_positive_rate * 100:.2f}%")
print(f"False Negative Rate: {false_negative_rate * 100:.2f}%")