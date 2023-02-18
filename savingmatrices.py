from Matrix import Matrix


def convert_matrix_to_string(matrix):
    matrix_rows_as_string = []
    for row in matrix.data:
        row_string = ' '.join([str(item) for item in row])
        matrix_rows_as_string.append(row_string)
    matrix_string = ','.join(matrix_rows_as_string)
    return matrix_string


def convert_string_to_matrix(string):
    array_rows = string.split(',')
    array = [[float(item) for item in row.split(' ')] for row in array_rows]
    matrix = Matrix(len(array), len(array[0]))
    matrix.data = array
    return matrix


def store_matrices_to_file(matrices, file_name):
    with open(file_name, 'w') as file:
        for matrix in matrices:
            matrix_string = convert_matrix_to_string(matrix)
            file.write(matrix_string + '\n')


def get_matrices_from_file(file_name):
    matrices = []
    with open(file_name, 'r') as file:
        for line in file:
            line.strip()
            matrix = convert_string_to_matrix(line)
            matrices.append(matrix)
    return matrices



# a = Matrix(4, 3)
# a.data = [[5, 5, 12.56],
#           [-1, 7, 18.076324],
#           [-0.12, 7432, 32],
#           [124, -135, 13.14]]
#
# b = Matrix(2, 2)
# b.data = [[5, 2],
#           [-1, 3]]
#
# store_matrices_to_file([a, b], 'a_matrix.txt')
# new_matrices = get_matrices_from_file('a_matrix.txt')
#
# for matrix in new_matrices:
#     print(matrix)
