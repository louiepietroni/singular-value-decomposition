import random


class Vector:
    def __init__(self, rows):
        self.rows = rows
        self.data = self.create_empty_vector()

    def __str__(self):
        return '\n' + '\n'.join([''.join(['{:4}'.format(str(round(item, 5))) for item in row]) for row in self.data])

    @staticmethod
    def create_vector_from_data(data):
        vector_rows = len(data)
        vector_data = [[item for item in row] for row in data]
        vector = Vector(vector_rows)
        vector.data = vector_data
        return vector

    def create_empty_vector(self):
        vector = [[0] for i in range(self.rows)]
        return vector

    def randomise(self):
        self.data = [[random.random()] for i in range(self.rows)]

    @staticmethod
    def multiply_by_matrix(vector, matrix):
        calculated_vector = Vector(vector.rows)
        for m in range(vector.rows):
            product = 0
            for i in range(vector.rows):
                item = matrix.data[m][i] * vector.data[i][0]
                product += item
            calculated_vector.data[m][0] = product
        return calculated_vector

    def get_magnitude(self):
        magnitude = 0
        for i in range(self.rows):
            item = self.data[i][0]
            magnitude += item ** 2
        magnitude = magnitude ** (1/2)
        return magnitude

    def get_magnitude_squared(self):
        magnitude = 0
        for i in range(self.rows):
            item = self.data[i][0]
            magnitude += item ** 2
        return magnitude

    def divide_by_scalar(self, scalar):
        for i in range(self.rows):
            self.data[i][0] = self.data[i][0] / scalar

    @staticmethod
    def get_eigen_value(vector, matrix):
        product = 0
        for i in range(vector.rows):
            item = matrix.data[0][i] * vector.data[i][0]
            product += item
        eigen_value = product / vector.data[0][0]
        return eigen_value

    @staticmethod
    def get_greatest_eigen_vector(matrix):
        vector = Vector(matrix.columns)
        vector.randomise()
        for i in range(25):
            iterated_vector = Vector.multiply_by_matrix(vector, matrix)
            iterated_vector_magnitude = iterated_vector.get_magnitude()
            iterated_vector.divide_by_scalar(iterated_vector_magnitude)
            vector = iterated_vector
        return vector
