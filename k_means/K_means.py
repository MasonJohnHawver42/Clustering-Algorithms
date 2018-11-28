import numpy as np

def find_euclidean_dis(a, point):
    point = np.reshape((np.tile(np.asarray(point), (a.shape[0]))), a.shape)
    squares = (a - point)
    squares = squares * squares

    sum_of_squares = np.zeros(squares.shape[0])
    for i in range(squares.shape[1]):
        sum_of_squares += squares[:, i]

    distances = np.sqrt(sum_of_squares)

    return distances


def k_means(data, k_points, max_iteratoion):

    done = False
    new_k_points = np.zeros(k_points.shape)
    distances = np.zeros((k_points.shape[0], data.shape[0]))
    k_means_output = np.zeros((k_points.shape[0], data.shape[0], data.shape[1]))

    iterations = 1
    while not done:
        for i in range(k_points.shape[0]):
            k_point = k_points[i]
            distances[i] = find_euclidean_dis(data, k_point)

        for i in range(distances.shape[0]):
            dis = distances[i]
            other_dis = np.zeros((distances.shape[0] + 1, distances.shape[1]))
            other_dis[1:] = distances
            other_dis = np.concatenate((other_dis[0:(i+1)], other_dis[i+2:]), axis=None)
            other_dis = np.reshape(other_dis, distances.shape)
            other_dis = other_dis[1:]
            ownership = other_dis > (np.reshape((np.tile(dis, other_dis.shape[0])), other_dis.shape))
            output = np.ones(ownership.shape[1])
            for j in ownership:
                output = output * j

            p = np.sum(output)
            output = data * np.reshape((np.repeat(output, data.shape[1])), data.shape)
            k_means_output[i] = output
            new_k_points[i] = np.nan_to_num((output.sum(axis=0) / p))

        if (np.array_equal(new_k_points, k_points)) or (max_iteratoion == iterations):
            done = True

        k_points = new_k_points
        iterations += 1
    return k_points, k_means_output


def make_k_points(num_k_points, data_dimension, data_min, data_max):

    choices = int(np.ceil((num_k_points ** (1 / data_dimension))))
    if choices == 1:
        choices = 3
    test = np.zeros(((np.power(choices, (data_dimension))), data_dimension))
    for i in range(test.shape[0]):
        output = np.ceil(((np.ones(data_dimension) * (i + 1)) / (np.power((np.ones(data_dimension) * choices), (np.arange(0, data_dimension))[::-1]))))
        test[i] = output % choices

    test = (test * ((data_max - data_min) / (choices - 1))) + data_min
    order = (np.sort(((np.stack((np.sum(test, axis=1), np.arange(0, test.shape[0])), axis=-1)).astype(np.int64)).view('i8,i8'), order=['f0'], axis=0)).view(np.int)
    test = test[order[:, 1]]
    order = ((np.arange(test.shape[0]).repeat(2) * ((np.arange(test.shape[0] * 2) % 2) == 0)) + (np.arange(test.shape[0] -1, -1, -1).repeat(2) * ((np.arange(test.shape[0] * 2) % 2) != 0)))[:test.shape[0]]
    test = test[order]
    return test[: num_k_points].astype(np.int64)
