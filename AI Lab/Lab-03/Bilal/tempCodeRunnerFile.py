plt.figure(figsize=(8, 6))
# plt.scatter(
#     robot_coordinates[:, 0], robot_coordinates[:, 1], c="blue", label="Robot Locations"
# )
# plt.plot(
#     reference_point[0],
#     reference_point[1],
#     marker="o",
#     markersize=10,
#     color="red",
#     label="Camera Reference Point",
# )
# plt.plot(
#     reference_point[0], robot_coordinates[0][0], marker=".", markersize=5, color="green"
# )

# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# plt.title("Distances between Camera and Robot Locations")
# plt.legend()

# for i, (x, y) in enumerate(robot_coordinates):
#     plt.annotate(
#         f"{distances[i]:.2f}",
#         (x, y),
#         textcoords="offset points",
#         xytext=(0, 10),
#         ha="center",
#     )

# plt.grid(True)
# plt.show()