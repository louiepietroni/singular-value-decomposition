class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        item = self.stack.pop()
        return item

    def get_top_item(self):
        return self.stack[-1]

    def is_empty(self):
        if len(self.stack) == 0:
            return True
        else:
            return False


def evaluate_term(operand_1, operand_2, operator):
    if operator == '+':
        return operand_1 + operand_2
    elif operator == '-':
        return operand_1 - operand_2
    elif operator == 'x':
        return operand_1 * operand_2
    elif operator == '/':
        return operand_1 / operand_2


def is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def get_precedence(operator):
    if operator == '+' or operator == '-':
        return 1
    if operator == 'x' or operator == '/':
        return 2
    return 0


def split_expression(expression):
    expression_list = []
    current_number = ''
    for token in expression:
        if is_number(token):
            current_number += token
        else:
            if current_number != '':
                expression_list.append(current_number)
                current_number = ''
            expression_list.append(token)
    if current_number != '':
        expression_list.append(current_number)
    return expression_list


def evaluate_expression(expression):
    expression = split_expression(expression)
    value_stack = Stack()
    operator_stack = Stack()

    while len(expression) > 0:
        next_token = expression.pop(0)
        if is_number(next_token):
            value_stack.push(float(next_token))
        elif next_token == '(':
            operator_stack.push(next_token)
        elif next_token == ')':
            while operator_stack.get_top_item() != '(':
                operator = operator_stack.pop()
                operand_2 = value_stack.pop()
                operand_1 = value_stack.pop()
                result = evaluate_term(operand_1, operand_2, operator)
                value_stack.push(result)
            operator_stack.pop()

        elif next_token in ['+', '-', 'x', '/']:
            while not operator_stack.is_empty() and get_precedence(operator_stack.get_top_item()) >= get_precedence(next_token):
                operator = operator_stack.pop()
                operand_2 = value_stack.pop()
                operand_1 = value_stack.pop()
                result = evaluate_term(operand_1, operand_2, operator)
                value_stack.push(result)
            operator_stack.push(next_token)

    while not operator_stack.is_empty():
        operator = operator_stack.pop()
        operand_2 = value_stack.pop()
        operand_1 = value_stack.pop()
        result = evaluate_term(operand_1, operand_2, operator)
        value_stack.push(result)
    return value_stack.pop()
