def euclidean_distance_tensor(point1, point2):
    return torch.sqrt(torch.sum((point1 - point2) ** 2))

# Modified KNN-TSP function
def knn_tsp_tensor(points, k):
    # Number of points (nodes)
    n = points.size(0)

    # Mark all nodes as unvisited
    visited = torch.zeros(n, dtype=torch.bool)
    # Initialize the tour with the first node
    tour = [0]
    visited[0] = True

    # Start at the first node
    current_node = 0

    # Iterate to build the tour
    while len(tour) < n:
        # Get the current node coordinates
        current_point = points[current_node]

        # Find the k nearest neighbors that haven't been visited
        neighbors = []
        for i in range(n):
            if not visited[i]:
                dist = euclidean_distance_tensor(current_point, points[i])
                neighbors.append((dist.item(), i))

        # Sort neighbors by distance and choose the nearest k
        neighbors.sort()
        k_nearest = neighbors[:k]

        # Pick the closest neighbor from the k nearest ones
        _, next_node = min(k_nearest)

        # Mark the node as visited
        visited[next_node] = True
        # Append the node to the tour
        tour.append(next_node)
        # Move to the next node
        current_node = next_node

    # Return to the starting point to complete the tour
    tour.append(0)

    # Prepare output in the desired format
    # Convert coordinates to space-separated string
    coordinates_string = " ".join([f"{coord.item()}" for point in points for coord in point])
    
    # Convert tour to space-separated string
    tour_string = " ".join(map(str, tour))
    
    # Format the output as required
    output = f"{coordinates_string} output {tour_string}"
    
    return output