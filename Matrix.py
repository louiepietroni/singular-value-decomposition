from Vector import Vector


class Matrix:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.data = self.create_empty_matrix()

    def __str__(self):
        return '\n' + '\n'.join([''.join(['{:6}'.format(str(round(item, 1))) for item in row]) for row in self.data])

    @staticmethod
    def create_matrix_from_data(data):
        matrix_rows = len(data)
        matrix_columns = len(data[0])
        matrix_data = [[item for item in row] for row in data]
        matrix = Matrix(matrix_rows, matrix_columns)
        matrix.data = matrix_data
        return matrix

    def create_empty_matrix(self):
        matrix = [[0 for i in range(self.columns)] for j in range(self.rows)]
        return matrix

    def multiply_by_scalar(self, scalar):
        for m in range(self.rows):
            for n in range(self.columns):
                self.data[m][n] = self.data[m][n] * scalar

    @staticmethod
    def subtract(matrix, second_matrix):
        calculated_matrix = Matrix(matrix.rows, matrix.columns)
        for m in range(matrix.rows):
            for n in range(matrix.columns):
                item = matrix.data[m][n] - second_matrix.data[m][n]
                calculated_matrix.data[m][n] = item
        return calculated_matrix

    @staticmethod
    def multiply_left_by_transpose(matrix):
        transposed_multiplied_matrix = Matrix(matrix.columns, matrix.columns)
        # print(matrix.columns)
        for m in range(matrix.columns):
            # print(m)
            for n in range(m, matrix.columns):
                product = 0
                for i in range(matrix.rows):
                    item = matrix.data[i][m] * matrix.data[i][n]
                    product += item
                transposed_multiplied_matrix.data[m][n] = product
                transposed_multiplied_matrix.data[n][m] = product
        return transposed_multiplied_matrix

    @staticmethod
    def multiply_right_vector_by_transpose(vector):
        calculated_matrix = Matrix(vector.rows, vector.rows)
        for m in range(vector.rows):
            for n in range(m, vector.rows):
                item = vector.data[m][0] * vector.data[n][0]
                calculated_matrix.data[m][n] = item
                calculated_matrix.data[n][m] = item
        return calculated_matrix

    def round_matrix(self, digits):
        for m in range(self.rows):
            for n in range(self.columns):
                self.data[m][n] = round(self.data[m][n], digits)

    def delete_row(self, row):
        del self.data[row]
        self.rows -= 1

    def delete_column(self, column):
        for m in range(self.rows):
            del self.data[m][column]
        self.columns -= 1

    @staticmethod
    def calculate_row_echelon_form(matrix):
        matrix_rows = matrix.rows
        matrix_columns = matrix.columns

        # echelon_matrix = Matrix(matrix_rows, matrix_columns)
        # echelon_matrix.data = [[item for item in row] for row in matrix.data]

        echelon_matrix = Matrix.create_matrix_from_data(matrix.data)

        r = matrix_rows - 1
        while r >= 0:
            if echelon_matrix.data[r][r] == 0:
                # row_r = echelon_matrix.data[r]
                # echelon_matrix.data[r] = echelon_matrix.data[matrix_rows - 1]
                # echelon_matrix.data[matrix_rows - 1] = row_r
                # for i in range(matrix_rows - 1):
                #     column_r = echelon_matrix.data[i][r]
                #     echelon_matrix.data[i][r] = echelon_matrix.data[i][matrix_rows - 1]
                #     echelon_matrix.data[i][matrix_rows - 1] = column_r
                # matrix_rows -= 1
                echelon_matrix.delete_row(r)
                echelon_matrix.delete_column(r)
                matrix_rows -= 1
                matrix_columns -= 1
            r -= 1
        # print(echelon_matrix, 'echelon matrix with 0 rows removed')

        r = 1
        while r < matrix_rows:
        # for r in range(1, matrix_rows):
            for c in range(r):
                multiple = echelon_matrix.data[r][c] / echelon_matrix.data[c][c]
                # print(echelon_matrix.data[r][c], echelon_matrix.data[c][c], multiple, 'multiple from / ')
                for i in range(c, matrix_columns):
                    new_value = echelon_matrix.data[c][i] * -multiple + echelon_matrix.data[r][i]
                    echelon_matrix.data[r][i] = new_value
                # print(echelon_matrix, 'updated with', r, c)

            all_zeros = True
            for i in range(matrix_columns):
                if echelon_matrix.data[r][i] != 0:
                    all_zeros = False
                    break
            if all_zeros:
                row_r = echelon_matrix.data[r]
                echelon_matrix.data[r] = echelon_matrix.data[matrix_rows - 1]
                echelon_matrix.data[matrix_rows - 1] = row_r
                matrix_rows -= 1
                # print(echelon_matrix, r, 'row was all 0 had to swap')
                r -= 1

            if echelon_matrix.data[r][r] == 0:
                greatest_value = 0
                greatest_value_index = r
                for i in range(r, matrix_rows):
                    if echelon_matrix.data[i][r] > greatest_value:
                        greatest_value = echelon_matrix.data[i][r]
                        greatest_value_index = i
                if greatest_value == 0:
                    echelon_matrix.delete_row(r)
                    echelon_matrix.delete_column(r)
                    matrix_rows -= 1
                    matrix_columns -= 1
                    # print('had to delete row and column as all 0 below in column', r)
                    r -= 1
                else:
                    row_r = echelon_matrix.data[r]
                    echelon_matrix.data[r] = echelon_matrix.data[greatest_value_index]
                    echelon_matrix.data[greatest_value_index] = row_r
                    # print('swapped row for one below with greatest value', r)
                    r -= 1

            r += 1

        # for i in range(matrix_columns, 1, -1):
        #     if echelon_matrix.data[i - 1] == echelon_matrix.data[i - 2]:
        #         echelon_matrix.data[i - 1] = [0 for j in range(matrix_columns)]

        # print(echelon_matrix, 'row echelon form')
        return echelon_matrix

    def get_rank(self):
        matrix = Matrix.calculate_row_echelon_form(self)
        matrix.round_matrix(5)
        matrix_rank = matrix.rows
        for r in range(matrix.rows, 0, -1):
            all_zeros = True
            for i in range(matrix.columns):
                if matrix.data[r - 1][i] != 0:
                    # print(matrix.data[r-1], r, i)
                    all_zeros = False
                    break
            if not all_zeros:
                break
            matrix_rank -= 1
        return matrix_rank

    def add_values_on_leading_diagonal(self, values):
        for i in range(len(values)):
            self.data[i][i] = values[i]

    @staticmethod
    def identity(size):
        identity_matrix = Matrix(size, size)
        identity_matrix.add_values_on_leading_diagonal([1 for _ in range(size)])
        return identity_matrix

    @staticmethod
    def create_matrix_from_vectors(vectors):
        matrix = Matrix(vectors[0].rows, len(vectors))
        for n in range(matrix.columns):
            for m in range(matrix.rows):
                matrix.data[m][n] = vectors[n].data[m][0]
        return matrix

    @staticmethod
    def multiply(matrix_a, matrix_b):
        matrix = Matrix(matrix_a.rows, matrix_b.columns)
        for m in range(matrix.rows):
            for n in range(matrix.columns):
                product = 0
                for i in range(matrix_a.columns):
                    item = matrix_a.data[m][i] * matrix_b.data[i][n]
                    product += item
                matrix.data[m][n] = product
        return matrix

    @staticmethod
    def inverse_of_leading_diagonal_matrix(original_matrix):
        matrix = Matrix.create_matrix_from_data(original_matrix.data)
        for i in range(matrix.rows):
            matrix.data[i][i] = 1 / matrix.data[i][i]
        return matrix

    @staticmethod
    def transpose(original_matrix):
        matrix = Matrix(original_matrix.columns, original_matrix.rows)
        for m in range(matrix.rows):
            for n in range(matrix.columns):
                matrix.data[m][n] = original_matrix.data[n][m]
        return matrix

    def make_integer(self):
        for m in range(self.rows):
            for n in range(self.columns):
                self.data[m][n] = int(self.data[m][n])

    def reduce_rows(self, r):
        self.data = self.data[:r]
        self.rows = r

    def reduce_columns(self, r):
        for i in range(self.rows):
            self.data[i] = self.data[i][:r]
        self.columns = r

    def reduce_diagonal(self, r):
        self.reduce_rows(r)
        self.reduce_columns(r)

    @staticmethod
    def split_three_dimension_array(array):
        array_rows = len(array)
        array_cols = len(array[0])
        matrix_a = Matrix(array_rows, array_cols)
        matrix_b = Matrix(array_rows, array_cols)
        matrix_c = Matrix(array_rows, array_cols)
        for m in range(array_rows):
            for n in range(array_cols):
                matrix_a_item = array[m][n][0]
                matrix_b_item = array[m][n][1]
                matrix_c_item = array[m][n][2]
                matrix_a.data[m][n] = matrix_a_item
                matrix_b.data[m][n] = matrix_b_item
                matrix_c.data[m][n] = matrix_c_item
        return matrix_a, matrix_b, matrix_c

    @staticmethod
    def concatenate_matrices_into_list(matrix_a, matrix_b, matrix_c):
        array = [[[item] for item in row] for row in matrix_a.data]
        array_rows = len(array)
        array_cols = len(array[0])
        for m in range(array_rows):
            for n in range(array_cols):
                matrix_b_item = matrix_b.data[m][n]
                matrix_c_item = matrix_c.data[m][n]
                array[m][n].append(matrix_b_item)
                array[m][n].append(matrix_c_item)
        return array

    def get_average(self):
        total = 0
        for m in range(self.rows):
            for n in range(self.columns):
                total += self.data[m][n]
        total_items = self.rows * self.columns
        average = total / total_items
        return average

    @staticmethod
    def sharpen(matrix):
        sharpened_matrix = Matrix(matrix.rows, matrix.columns)
        for m in range(1, sharpened_matrix.rows - 1):
            for n in range(1, sharpened_matrix.columns - 1):
                item_top = matrix.data[m-1][n] * -1
                item_left = matrix.data[m][n-1] * -1
                item_centre = matrix.data[m][n] * 5
                item_right = matrix.data[m][n+1] * -1
                item_bottom = matrix.data[m+1][n] * -1
                sharpened_item = item_top + item_left + item_centre + item_right + item_bottom
                sharpened_matrix.data[m][n] = sharpened_item
        return sharpened_matrix

    @staticmethod
    def filter(matrix):
        filtered_matrix = Matrix(matrix.rows, matrix.columns)
        for m in range(1, filtered_matrix.rows - 1):
            for n in range(1, filtered_matrix.columns - 1):
                item_top = matrix.data[m - 1][n]
                item_left = matrix.data[m][n - 1]
                item_centre = matrix.data[m][n]
                item_right = matrix.data[m][n + 1]
                item_bottom = matrix.data[m + 1][n]
                filter_items = [item_top, item_left, item_centre, item_right, item_bottom]
                filter_items.sort()
                median_filter_item = filter_items[2]
                filtered_matrix.data[m][n] = median_filter_item
        return filtered_matrix

    @staticmethod
    def detect_edges(matrix):
        edge_matrix = Matrix(matrix.rows, matrix.columns)
        for m in range(1, edge_matrix.rows - 1):
            for n in range(1, edge_matrix.columns - 1):
                item_top_left = matrix.data[m - 1][n - 1] * -1
                item_top_centre = matrix.data[m - 1][n] * -1
                item_top_right = matrix.data[m - 1][n + 1] * -1
                item_centre_left = matrix.data[m][n - 1] * -1
                item_centre_centre = matrix.data[m][n] * 8
                item_centre_right = matrix.data[m][n + 1] * -1
                item_bottom_left = matrix.data[m + 1][n - 1] * -1
                item_bottom_centre = matrix.data[m + 1][n] * -1
                item_bottom_right = matrix.data[m + 1][n + 1] * -1

                edge_item = item_top_left + item_top_centre + item_top_right + item_centre_left + item_centre_centre + item_centre_right + item_bottom_left + item_bottom_centre + item_bottom_right

                if edge_item < 120:
                    edge_item = 0
                else:
                    edge_item = 255

                edge_matrix.data[m][n] = edge_item
        return edge_matrix

    @staticmethod
    def sharpen_edges(matrix, edge_matrix):
        sharpened_matrix = Matrix(matrix.rows, matrix.columns)
        for m in range(1, sharpened_matrix.rows - 1):
            for n in range(1, sharpened_matrix.columns - 1):
                if edge_matrix.data[m][n] == 255:
                    item_top = matrix.data[m - 1][n] * -1
                    item_left = matrix.data[m][n - 1] * -1
                    item_centre = matrix.data[m][n] * 5
                    item_right = matrix.data[m][n + 1] * -1
                    item_bottom = matrix.data[m + 1][n] * -1
                    sharpened_item = item_top + item_left + item_centre + item_right + item_bottom
                    sharpened_matrix.data[m][n] = sharpened_item
                else:
                    sharpened_matrix.data[m][n] = matrix.data[m][n]
        return sharpened_matrix

    def invert(self):
        for m in range(self.rows):
            for n in range(self.columns):
                self.data[m][n] = 255 - self.data[m][n]

    def black_or_white(self, limit):
        for m in range(self.rows):
            for n in range(self.columns):
                if self.data[m][n] < limit:
                    self.data[m][n] = 0
                else:
                    self.data[m][n] = 255

    def strip(self):
        minimum_row = 0
        for row in self.data:
            if sum(row) == 0:
                minimum_row += 1
            else:
                break

        maximum_row = self.rows
        for row in self.data[::-1]:
            if sum(row) == 0:
                maximum_row -= 1
            else:
                break

        self.rows = maximum_row - minimum_row
        self.data = self.data[minimum_row:maximum_row]

        transposed = Matrix.transpose(self)

        minimum_row = 0
        for row in transposed.data:
            if sum(row) == 0:
                minimum_row += 1
            else:
                break

        maximum_row = transposed.rows
        for row in transposed.data[::-1]:
            if sum(row) == 0:
                maximum_row -= 1
            else:
                break

        for row_index, row in enumerate(self.data):
            self.data[row_index] = row[minimum_row:maximum_row]

        self.columns = maximum_row - minimum_row

    def pad(self, rows, columns):
        rows_to_add_to_top = int(rows * (1/2))
        rows_to_add_to_bottom = rows - rows_to_add_to_top
        zero_row = [0 for _ in range(self.columns)]
        rows_for_top = [zero_row for _ in range(rows_to_add_to_top)]
        rows_for_bottom = [zero_row for _ in range(rows_to_add_to_bottom)]
        self.data = rows_for_top + self.data + rows_for_bottom
        self.rows += rows

        columns_to_add_to_left = int(columns * (1/2))
        columns_to_add_to_right = columns - columns_to_add_to_left
        zero_columns_for_left = [0 for _ in range(columns_to_add_to_left)]
        zero_columns_for_right = [0 for _ in range(columns_to_add_to_right)]

        for index, row in enumerate(self.data):
            self.data[index] = zero_columns_for_left + self.data[index] + zero_columns_for_right
        self.columns += columns

    @staticmethod
    def resize(matrix, size):
        reduced_matrix = Matrix(size, size)
        factor = int(matrix.rows / reduced_matrix.rows)
        for m in range(reduced_matrix.rows):
            for n in range(reduced_matrix.columns):
                total = 0
                for i in range(m * factor, (m + 1) * factor):
                    for j in range(n * factor, (n + 1) * factor):
                        total += matrix.data[i][j]
                mean = total / (factor) ** 2
                reduced_matrix.data[m][n] = int(mean)
        return reduced_matrix

    def calculate_centre_of_mass(self):
        x_total = 0
        x_total_items = 0
        for row in self.data:
            for index, item in enumerate(row):
                if item != 0:
                    x_total_items += 1
                    x_total += index
        if x_total_items != 0:
            x_mean = x_total / x_total_items
        else:
            x_mean = 0

        transposed = Matrix.transpose(self)
        y_total = 0
        y_total_items = 0
        for row in transposed.data:
            for index, item in enumerate(row):
                if item != 0:
                    y_total_items += 1
                    y_total += index
        if y_total_items != 0:
            y_mean = y_total / y_total_items
        else:
            y_mean = 0
        return x_mean, y_mean

    def pad_edges(self, top, bottom, left, right):
        zero_row = [0 for _ in range(self.columns)]
        rows_for_top = [zero_row for _ in range(top)]
        rows_for_bottom = [zero_row for _ in range(bottom)]
        self.data = rows_for_top + self.data + rows_for_bottom
        self.rows += top + bottom

        zero_columns_for_left = [0 for _ in range(left)]
        zero_columns_for_right = [0 for _ in range(right)]

        for index, row in enumerate(self.data):
            self.data[index] = zero_columns_for_left + self.data[index] + zero_columns_for_right
        self.columns += left + right

    def get_full_left_columns(self):
        transposed = Matrix.transpose(self)
        left_columns = 0
        for row in transposed.data:
            if sum(row) != 0:
                left_columns += 1
            else:
                break
        return left_columns

    def get_empty_columns_after_column(self, index):
        transposed = Matrix.transpose(self)
        empty_columns = 0
        for row in transposed.data[index:]:
            if sum(row) == 0:
                empty_columns += 1
            else:
                break
        return empty_columns

    @staticmethod
    def get_first_columns(matrix, columns):
        new_matrix = Matrix(matrix.rows, columns)
        new_matrix.data = [row[:columns] for row in matrix.data]
        return new_matrix

    @staticmethod
    def remove_left_columns(matrix, index):
        new_matrix = Matrix(matrix.rows, matrix.columns - index)
        new_matrix.data = [row[index:] for row in matrix.data]
        return new_matrix

    def get_full_top_rows(self):
        top_rows = 0
        for row in self.data:
            if sum(row) != 0:
                top_rows += 1
            else:
                break
        return top_rows

    @staticmethod
    def get_first_rows(matrix, rows):
        new_matrix = Matrix(rows, matrix.columns)
        new_matrix.data = [row for row in matrix.data[:rows]]
        return new_matrix

    def get_empty_rows_after_row(self, index):
        empty_rows = 0
        for row in self.data[index:]:
            if sum(row) == 0:
                empty_rows += 1
            else:
                break
        return empty_rows

    @staticmethod
    def remove_bottom_rows(matrix, index):
        new_matrix = Matrix(matrix.rows - index, matrix.columns)
        new_matrix.data = [row for row in matrix.data[index:]]
        return new_matrix

    @staticmethod
    def flatten_along_column(matrix):
        new_matrix = Matrix(matrix.rows * matrix.columns, 1)
        new_matrix_data = []
        for column in range(matrix.columns):
            for row in matrix.rows:
                new_matrix_data.append([row[column]])
        new_matrix.data = new_matrix_data
        return new_matrix

    @staticmethod
    def flatten_along_rows(matrix):
        new_matrix = Matrix(1, matrix.rows * matrix.columns)
        new_matrix_data = [[]]
        for row in matrix.data:
            for item in row:
                new_matrix_data[0].append(item)
        new_matrix.data = new_matrix_data
        return new_matrix

    @staticmethod
    def concatenate_matrices_side(matrix_a, matrix_b):
        new_matrix = Matrix(matrix_a.rows, matrix_a.columns + matrix_b.columns)
        new_matrix_data = [row_a + row_b for row_a, row_b in zip(matrix_a.data, matrix_b.data)]
        new_matrix.data = new_matrix_data
        return new_matrix




